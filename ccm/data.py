"""Immutable, document-isolated token/segment manifests and exact-token batches."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
import torch
from .contracts import (ROLES, require, fresh_dir, write_json, read_json, file_hash,
                        digest_json, load_budget, VERSION)
from .keys import position_keys


def source_id(revision, shard, row):
    # Structured serialization avoids ambiguous string concatenation.
    return digest_json([revision, "en", shard, int(row)])


def verify_sources(root, manifest, revision):
    require(len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
            "Pin the official dataset commit SHA")
    text = Path(manifest).read_text()
    require(revision in text, "Dataset revision missing from trusted manifest")
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sha, size, rel = line.split()
        if not rel.startswith("en/"):
            continue
        path = (Path(root)/rel).resolve()
        require(path.is_relative_to(Path(root).resolve()), "Manifest path traversal")
        require(path.stat().st_size == int(size) and file_hash(path) == sha,
                f"Raw source checksum mismatch: {rel}")
        rows.append((rel, path))
    require(rows, "No verified English shards")
    return sorted(rows)


def role_for(content_hash, quotas, seed=42):
    point = int(hashlib.sha256(f"{seed}:{content_hash}".encode()).hexdigest(), 16)
    point %= sum(quotas.values())
    for role in ROLES:
        if point < quotas[role]:
            return role
        point -= quotas[role]
    raise RuntimeError("Unreachable role assignment")


def cuts(length, offset, role, budget):
    """Union of phase quota grids; fixed BEFORE common pretraining.

    Common data is compile -> adapt -> common_other before whole-batch shuffle.
    Adapt uses its own batch grid as well; cuts must satisfy BOTH grids.
    """
    q = budget.quotas()
    common_offsets = dict(compile=0, adapt=q["compile"], common_other=q["compile"]+q["adapt"])
    grids = []
    if role in common_offsets:
        grids.append(common_offsets[role])
    if role in ("adapt", "continue"):
        grids.append(0)
    pos = 0
    while pos < length:
        size = min(budget.segment_length, length-pos)
        for start in grids:
            size = min(size, budget.batch_tokens-((start+offset+pos) % budget.batch_tokens))
        yield pos, pos+size
        pos += size


def build_manifest(documents, tokenizer, output, budget, provenance, engineering=False):
    """documents yields (doc_id, exact text); stream to uint32 token files.

    Exact duplicate source rows are retained within their deterministic role.
    Full source scanning is avoided after every quota is filled. Duplicate
    diagnostics explicitly cover scanned rows, not unseen source files.
    """
    load_budget(budget.to_dict(), engineering)
    require(tokenizer.eos_token_id is not None, "Tokenizer has no document EOS")
    out = fresh_dir(output)
    quotas = budget.quotas()
    used = dict.fromkeys(ROLES, 0)
    segment_counts = dict.fromkeys(ROLES, 0)
    db = sqlite3.connect(out/"documents.sqlite")
    db.execute("CREATE TABLE content (hash TEXT PRIMARY KEY, role TEXT, occurrences INTEGER)")
    db.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, hash TEXT, role TEXT)")
    scanned = 0
    with ExitStack() as stack:
        token_files = {r: stack.enter_context((out/f"{r}.tokens").open("xb")) for r in ROLES}
        indices = {r: stack.enter_context((out/f"{r}.segments.jsonl").open("x")) for r in ROLES}
        for doc_id, text in documents:
            if all(used[r] == quotas[r] for r in ROLES):
                break
            require(isinstance(text, str), f"Non-text source document {doc_id}")
            ch = hashlib.sha256(text.encode("utf8")).hexdigest()
            role = role_for(ch, quotas)
            scanned += 1
            db.execute("INSERT INTO content VALUES (?,?,1) ON CONFLICT(hash) DO UPDATE SET occurrences=occurrences+1", (ch, role))
            if used[role] == quotas[role]:
                continue
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]+[tokenizer.eos_token_id]
            ids = ids[:quotas[role]-used[role]]
            require(all(0 <= x < 2**31 for x in ids), "Token ID outside supported range")
            db.execute("INSERT INTO documents VALUES (?,?,?)", (doc_id, ch, role))
            for a, b in cuts(len(ids), used[role], role, budget):
                row = dict(doc_id=doc_id, content_hash=ch, offset=used[role]+a,
                           length=b-a, segment_id=f"{role}:{segment_counts[role]}")
                indices[role].write(json.dumps(row, separators=(",", ":"))+"\n")
                segment_counts[role] += 1
            token_files[role].write(np.asarray(ids, dtype="<u4").tobytes())
            used[role] += len(ids)
            if scanned % 1000 == 0:
                db.commit()
                print(json.dumps(dict(scanned_documents=scanned, role_tokens=used)), flush=True)
    db.commit()
    duplicates = db.execute("SELECT COUNT(*), COALESCE(SUM(occurrences-1),0) FROM content WHERE occurrences>1").fetchone()
    db.close()
    require(used == quotas, f"Insufficient verified source data: {used}, required {quotas}")
    files = {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}
    meta = dict(version=VERSION, engineering=engineering, budget=budget.to_dict(),
                provenance=provenance, special_ids=sorted(tokenizer.all_special_ids),
                eos_id=tokenizer.eos_token_id, quotas=quotas, segments=segment_counts,
                scanned_documents=scanned, duplicate_groups=duplicates[0], duplicate_extra_rows=duplicates[1],
                split_seed=42, selection="content-hash weighted immutable role, exact prefixes",
                batch_order="shuffle preconstructed whole optimizer batches only", files=files)
    meta["manifest_hash"] = digest_json(meta)
    write_json(out/"manifest.json", meta)
    return meta


class Corpus:
    def __init__(self, path, verify=True):
        self.path = Path(path)
        self.meta = read_json(self.path/"manifest.json")
        d = dict(self.meta)
        expected = d.pop("manifest_hash")
        require(digest_json(d) == expected, "Manifest metadata corrupted")
        if verify:
            for name, sha in self.meta["files"].items():
                require(file_hash(self.path/name) == sha, f"Corpus checksum mismatch: {name}")
        self.budget = load_budget(self.meta["budget"], self.meta["engineering"])
        require(self.meta["version"] == VERSION, "Unsupported corpus version")
        require(self.meta["quotas"] == self.budget.quotas(), "Manifest quotas disagree with locked budget")
        required = {f"{r}.{suffix}" for r in ROLES for suffix in ("tokens", "segments.jsonl")} | {"documents.sqlite"}
        require(required <= set(self.meta["files"]), "Missing required corpus file checksum")
        for role in ROLES:
            require((self.path/f"{role}.tokens").stat().st_size == 4*self.meta["quotas"][role],
                    f"Wrong token-file byte length: {role}")
        self._tokens = {r: np.memmap(self.path/f"{r}.tokens", dtype="<u4", mode="r") for r in ROLES}

    def segments(self, role):
        require(role in ROLES, "Unknown data role")
        with (self.path/f"{role}.segments.jsonl").open() as f:
            end = 0
            for line in f:
                row = json.loads(line)
                require(row["offset"] == end and 0 < row["length"] <= self.budget.segment_length,
                        "Invalid/gapped segment manifest")
                end += row["length"]
                require(end <= len(self._tokens[role]), "Segment runs past token-file end")
                row["tokens"] = self._tokens[role][row["offset"]:end]
                yield row
            require(end == self.meta["quotas"][role], "Segment quota mismatch")

    def optimizer_batches(self, phase, seed):
        roles = {"common": ("compile", "adapt", "common_other"), "stage1": ("adapt",),
                 "stage2": ("continue",)}[phase]
        # Indices are modest (~millions of rows); token payload remains mmap-backed.
        batches, batch, n = [], [], 0
        for role in roles:
            for row in self.segments(role):
                batch.append(row)
                n += len(row["tokens"])
                require(n <= self.budget.batch_tokens, "Manifest crosses an optimizer boundary")
                if n == self.budget.batch_tokens:
                    batches.append(batch)
                    batch, n = [], 0
        require(n == 0, "Incomplete optimizer batch")
        order = np.random.default_rng(seed).permutation(len(batches))
        for i in order:
            yield batches[i]


def collate(rows, special_ids, vocab=None):
    """Padding/bucketing: one segment per row, hence no cross-segment attention.

    This is the explicit correctness-first backend permitted by the contract;
    does not concatenate short documents into ordinary causal attention.
    """
    require(len(rows) > 0, "Empty physical microbatch")
    length = max(len(r["tokens"]) for r in rows)
    shape = (len(rows), length)
    ids = torch.zeros(shape, dtype=torch.long)
    mask = torch.zeros(shape, dtype=torch.bool)
    targets = torch.full(shape, -100, dtype=torch.long)
    slots = torch.full(shape, -1, dtype=torch.int32)
    eligible = torch.zeros(shape, dtype=torch.bool)
    for i, row in enumerate(rows):
        x = np.asarray(row["tokens"], dtype=np.int64)
        n = len(x)
        ids[i, :n] = torch.from_numpy(x.copy())
        mask[i, :n] = True
        targets[i, :n-1] = torch.from_numpy(x[1:].copy())
        if vocab:
            s, e = vocab.route(x, special_ids)
            slots[i, :n] = torch.from_numpy(s)
        else:
            _, e = position_keys(x, special_ids)
        eligible[i, :n] = torch.from_numpy(e)
    return dict(input_ids=ids, attention_mask=mask, targets=targets, slots=slots,
                eligible=eligible, position_ids=torch.arange(length).expand(len(rows), -1))


def microbatches(rows, max_segments):
    require(max_segments > 0, "Microbatch size must be positive")
    ordered = sorted(rows, key=lambda r: (len(r["tokens"]), r["segment_id"]))
    for i in range(0, len(ordered), max_segments):
        yield ordered[i:i+max_segments]


def validate_splits(corpus):
    """Disk-side validation avoids materializing all document identities."""
    db = sqlite3.connect(f"file:{corpus.path/'documents.sqlite'}?mode=ro", uri=True)
    bad = db.execute("SELECT hash FROM documents GROUP BY hash HAVING COUNT(DISTINCT role)>1 LIMIT 1").fetchone()
    require(bad is None, "Exact-text leakage across roles")
    counts = dict(db.execute("SELECT role,COUNT(*) FROM documents GROUP BY role"))
    db.close()
    require(set(counts) == set(ROLES), "Missing role documents")
    return counts
