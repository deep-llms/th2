"""28L data extension: immutable historical prefix, new disjoint locked holdout.

Historical payloads are referenced by absolute path AND verified manifest hash;
they are never edited or re-tokenized. Extensions are selected in three ordered
source passes, reserving val28 before continuation and common. SQLite bounds
identity/deduplication RAM. A partial preparation has no manifest.json.
"""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import numpy as np
from .contracts import Budget, require, fresh_dir, write_json, read_json, file_hash, digest_json, VERSION
from .data import Corpus, validate_splits
from .studies import SCALEUP, SCALEUP_BUDGET, require_vocabulary

EXTENSIONS = ("val28", "continue_extension", "common_extension")


def frozen_batch_orders(historical_budget, budget):
    result = {}
    for seed in (1017, 1029, 1043):
        result[str(seed)] = {}
        for phase, field in (("common", "common_steps"), ("stage1", "adapt_steps"), ("stage2", "continue_steps")):
            old, total = getattr(historical_budget, field), getattr(budget, field)
            result[str(seed)][phase] = dict(
                historical=np.random.default_rng(seed).permutation(old).tolist(),
                extension=np.random.default_rng(seed).permutation(total-old).tolist())
    return result


def extension_quotas(historical, budget):
    b = historical.budget
    for field in ("batch_tokens", "adapt_steps", "compile_tokens", "dev_tokens", "val_tokens",
                  "segment_length", "slots"):
        require(getattr(b, field) == getattr(budget, field), f"Historical invariant changed: {field}")
    q = dict(val28=budget.val_tokens,
             continue_extension=(budget.continue_steps-b.continue_steps)*b.batch_tokens,
             common_extension=(budget.common_steps-b.common_steps)*b.batch_tokens)
    require(all(n > 0 for n in q.values()), "Scale-up must extend both optimization streams")
    return q


def prepare_scaleup(documents_factory, tokenizer, historical, vocab, output, provenance,
                    budget=SCALEUP_BUDGET, engineering=False):
    require(historical.meta["engineering"] == engineering, "Historical engineering mode mismatch")
    require(engineering or budget == SCALEUP_BUDGET, "Wrong 28L scientific budget")
    validate_splits(historical)
    require_vocabulary(historical, vocab)
    require(len(vocab.keys) == budget.slots, "Historical vocabulary has the wrong capacity")
    for key in ("tokenizer", "dataset", "language", "revision", "raw_manifest_hash"):
        require(provenance.get(key) == historical.meta["provenance"].get(key), f"Source provenance changed: {key}")
    require(sorted(tokenizer.all_special_ids) == historical.meta["special_ids"]
            and tokenizer.eos_token_id == historical.meta["eos_id"], "Tokenizer special IDs changed")
    quotas = extension_quotas(historical, budget)
    out = fresh_dir(output)
    counts, selected = {}, {}
    with closing(sqlite3.connect(out/"documents.sqlite")) as db:
        db.execute("CREATE TABLE content (hash TEXT PRIMARY KEY, role TEXT NOT NULL)")
        db.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, hash TEXT NOT NULL, role TEXT NOT NULL)")
        db.execute("ATTACH DATABASE ? AS historical", (str(historical.path/"documents.sqlite"),))
        db.execute("INSERT INTO documents SELECT id,hash,role FROM historical.documents")
        db.execute("INSERT INTO content SELECT DISTINCT hash,role FROM historical.documents")
        db.commit()
        db.execute("DETACH DATABASE historical")
        for role in EXTENSIONS:
            used = segments = scanned = 0
            with (out/f"{role}.tokens").open("xb") as tf, (out/f"{role}.segments.jsonl").open("x") as sf:
                for doc_id, text in documents_factory():
                    if used == quotas[role]:
                        break
                    require(isinstance(text, str), "Non-text source document")
                    ch = hashlib.sha256(text.encode("utf8")).hexdigest()
                    scanned += 1
                    prior = db.execute("SELECT role FROM content WHERE hash=?", (ch,)).fetchone()
                    if prior is not None and prior[0] != role:
                        continue
                    # Detect changed text for a reused source ID, even if its
                    # content hash is different. Never silently alias documents.
                    identity = db.execute("SELECT hash,role FROM documents WHERE id=?", (doc_id,)).fetchone()
                    require(identity is None, "Repeated or changed source document identity")
                    ids = tokenizer(text, add_special_tokens=False)["input_ids"]+[tokenizer.eos_token_id]
                    ids = ids[:quotas[role]-used]
                    require(all(type(x) is int and 0 <= x < 2**31 for x in ids), "Invalid token IDs")
                    db.execute("INSERT OR IGNORE INTO content VALUES (?,?)", (ch, role))
                    db.execute("INSERT INTO documents VALUES (?,?,?)", (doc_id, ch, role))
                    pos = 0
                    while pos < len(ids):
                        size = min(budget.segment_length, len(ids)-pos)
                        if role != "val28":
                            size = min(size, budget.batch_tokens-(used+pos) % budget.batch_tokens)
                        row = dict(doc_id=doc_id, content_hash=ch, offset=used+pos,
                                   length=size, segment_id=f"{role}:{segments}")
                        sf.write(json.dumps(row, separators=(",", ":"))+"\n")
                        segments += 1
                        pos += size
                    tf.write(np.asarray(ids, dtype="<u4").tobytes())
                    used += len(ids)
                    if scanned % 1000 == 0:
                        db.commit()
                        print(json.dumps(dict(role=role, tokens=used, required=quotas[role])), flush=True)
            db.commit()
            require(used == quotas[role], f"Insufficient source data for {role}: {used}/{quotas[role]}")
            counts[role], selected[role] = segments, used
    # Small copied index keeps compiler artifact metadata compatible. Token
    # payload and actual compile iteration still come from checked historical data.
    shutil.copy2(historical.path/"compile.segments.jsonl", out/"compile.segments.jsonl")
    files = {p.name: file_hash(p) for p in out.iterdir() if p.is_file()}
    meta = dict(version=VERSION, study=SCALEUP, engineering=engineering, budget=budget.to_dict(),
                provenance=provenance, special_ids=historical.meta["special_ids"], eos_id=historical.meta["eos_id"],
                historical_path=str(historical.path.resolve()), historical_manifest_hash=historical.meta["manifest_hash"],
                historical_vocabulary_hash=vocab.hash, ordered_mapping_hash=vocab.mapping_hash,
                extension_quotas=selected, extension_segments=counts, files=files,
                quotas=dict(compile=budget.compile_tokens, adapt=budget.adapt_steps*budget.batch_tokens,
                            dev=budget.dev_tokens, val=budget.val_tokens, **selected),
                allocation_order=list(EXTENSIONS), historical_val_reused=False,
                batch_order="historical seed-shuffled prefix, then separately seed-shuffled extension",
                optimizer_batch_orders=frozen_batch_orders(historical.budget, budget),
                holdout="val maps only to newly reserved val28; historical val is excluded")
    meta["manifest_hash"] = digest_json(meta)
    write_json(out/"manifest.json", meta)
    validate_scaleup(ScaleupCorpus(out))
    write_json(out/"complete.json", dict(success=True, manifest_hash=meta["manifest_hash"]))
    return meta


class ScaleupCorpus:
    def __init__(self, path, verify=True):
        self.path = Path(path)
        self.meta = read_json(self.path/"manifest.json")
        d = dict(self.meta)
        require(digest_json({k: v for k, v in d.items() if k != "manifest_hash"}) == d["manifest_hash"],
                "Scale-up manifest metadata corrupted")
        require(d["study"] == SCALEUP and d["version"] == VERSION, "Wrong scale-up version")
        self.historical = Corpus(d["historical_path"], verify=verify)
        require(self.historical.meta["manifest_hash"] == d["historical_manifest_hash"], "Historical manifest changed")
        self.budget = Budget(**d["budget"])
        require(d["engineering"] == self.historical.meta["engineering"], "Engineering mode changed")
        require(d["engineering"] or self.budget == SCALEUP_BUDGET, "Wrong 28L budget")
        require(d["extension_quotas"] == extension_quotas(self.historical, self.budget), "Wrong extension quotas")
        require(d["optimizer_batch_orders"] == frozen_batch_orders(self.historical.budget, self.budget),
                "Frozen batch orders differ from historical prefix/extension seed convention")
        expected_quotas = dict(compile=self.budget.compile_tokens, adapt=self.budget.adapt_steps*self.budget.batch_tokens,
                               dev=self.budget.dev_tokens, val=self.budget.val_tokens, **d["extension_quotas"])
        require(d["quotas"] == expected_quotas, "Scale-up role quotas changed")
        require(d["special_ids"] == self.historical.meta["special_ids"] and d["eos_id"] == self.historical.meta["eos_id"],
                "Scale-up token controls changed")
        for key in ("tokenizer", "dataset", "language", "revision", "raw_manifest_hash"):
            require(d["provenance"].get(key) == self.historical.meta["provenance"].get(key), "Scale-up provenance changed")
        required = {f"{r}.{suffix}" for r in EXTENSIONS for suffix in ("tokens", "segments.jsonl")}
        require(required | {"documents.sqlite", "compile.segments.jsonl"} <= set(d["files"]), "Missing extension checksum")
        require(d["files"]["compile.segments.jsonl"] == self.historical.meta["files"]["compile.segments.jsonl"],
                "Historical compile index changed")
        if verify:
            for name, sha in d["files"].items():
                require((self.path/name).resolve().is_relative_to(self.path.resolve()), "Manifest path traversal")
                require(file_hash(self.path/name) == sha, f"Scale-up checksum mismatch: {name}")
        self._tokens = {}
        for r in EXTENSIONS:
            require((self.path/f"{r}.tokens").stat().st_size == d["extension_quotas"][r]*4,
                    f"Wrong extension bytes: {r}")
            self._tokens[r] = np.memmap(self.path/f"{r}.tokens", dtype="<u4", mode="r")

    def segments(self, role):
        if role in ("compile", "adapt", "dev"):
            yield from self.historical.segments(role)
            return
        if role == "val":
            role = "val28"
        require(role in EXTENSIONS, "Unknown scale-up role; historical val is not the 28L holdout")
        end = count = 0
        with (self.path/f"{role}.segments.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                require(row["offset"] == end and 0 < row["length"] <= self.budget.segment_length,
                        "Invalid extension segment")
                require(row["segment_id"] == f"{role}:{count}", "Noncanonical extension segment ID")
                end += row["length"]
                require(end <= len(self._tokens[role]), "Extension segment exceeds payload")
                row["tokens"] = self._tokens[role][row["offset"]:end]
                yield row
                count += 1
        require(end == self.meta["extension_quotas"][role] and count == self.meta["extension_segments"][role],
                "Extension quota/segment count mismatch")

    def optimizer_batches(self, phase, seed):
        # Never globally reshuffle old+new batches: that would destroy the
        # already-consumed historical prefix for each data-order seed.
        require(str(seed) in self.meta["optimizer_batch_orders"], "Data-order seed is not frozen in the manifest")
        order = self.meta["optimizer_batch_orders"][str(seed)][phase]
        yield from self.historical.optimizer_batches(phase, seed)
        if phase == "stage1":
            return
        role = {"common": "common_extension", "stage2": "continue_extension"}[phase]
        batches, batch, n = [], [], 0
        for row in self.segments(role):
            batch.append(row)
            n += len(row["tokens"])
            require(n <= self.budget.batch_tokens, "Extension crosses optimizer boundary")
            if n == self.budget.batch_tokens:
                batches.append(batch)
                batch, n = [], 0
        require(n == 0, "Incomplete extension optimizer batch")
        require(len(order["extension"]) == len(batches), "Wrong frozen extension batch count")
        for i in order["extension"]:
            yield batches[i]


def validate_scaleup(corpus):
    validate_splits(corpus.historical)
    with closing(sqlite3.connect(f"file:{corpus.path/'documents.sqlite'}?mode=ro", uri=True)) as db:
        db.execute("ATTACH DATABASE ? AS historical", (str(corpus.historical.path/"documents.sqlite"),))
        require(db.execute("SELECT id,hash,role FROM historical.documents EXCEPT SELECT id,hash,role FROM documents LIMIT 1").fetchone()
                is None, "Historical document exclusions missing or changed")
        require(db.execute("SELECT hash FROM documents GROUP BY hash HAVING COUNT(DISTINCT role)>1 LIMIT 1").fetchone()
                is None, "Content leakage across historical/extension roles")
        for role in EXTENSIONS:
            for row in corpus.segments(role):
                require(db.execute("SELECT hash,role FROM documents WHERE id=?", (row["doc_id"],)).fetchone()
                        == (row["content_hash"], role), "Extension identity missing or wrong role")
    return dict(success=True, historical_manifest_hash=corpus.historical.meta["manifest_hash"],
                manifest_hash=corpus.meta["manifest_hash"], ordered_mapping_hash=corpus.meta["ordered_mapping_hash"])
