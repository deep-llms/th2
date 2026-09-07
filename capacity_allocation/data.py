"""Offline, frozen document splits and contiguous token packs; no Hub requests."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_fraction(key, salt):
    return int(digest(salt + ":" + key)[:16], 16) / 2**64


def document_keys(row, cluster_field=None):
    text_hash = digest(row["text"])
    provenance = [str(row.get(k) or "") for k in ("source", "url", "timestamp")]
    stable_id = digest(json.dumps(provenance + [text_hash], ensure_ascii=False))
    cluster = row.get(cluster_field) if cluster_field else None
    if cluster_field and cluster is None:
        raise ValueError(f"Missing configured duplicate cluster field {cluster_field}")
    split_key = "cluster:" + str(cluster) if cluster_field else "text:" + text_hash
    return stable_id, text_hash, split_key


def rows_from_shard(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)
    else:
        raise ValueError("Source shards must be parquet or JSONL")


def prepare(source_manifest, tokenizer, output, *, sequence_length=2048, seed=0,
            sample_fraction=1.0, validation_fraction=.004, test_fraction=.004,
            cluster_field=None, tokenizer_provenance=None, minimum_tokens=None):
    """Scan ALL explicitly selected shards; hash sampling is not prefix sampling.

    Manifest `shards` contains path/sha256 entries, paths relative to manifest.
    Exact duplicates are dropped globally; split assignment uses cluster IDs if
    supplied, otherwise content hashes (stronger exact-leakage protection than
    provenance IDs). Near-duplicate detection is explicitly a separate audit.
    """
    if (sequence_length < 2 or not 0 < sample_fraction <= 1 or
            not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or
            validation_fraction + test_fraction >= 1):
        raise ValueError("Invalid packing/sampling/split settings")
    if tokenizer.eos_token_id is None or len(tokenizer) > 2**32:
        raise ValueError("Tokenizer must have EOS and fit uint32")
    source_manifest, output = Path(source_manifest).resolve(), Path(output)
    source = json.loads(source_manifest.read_text())
    if not re.fullmatch(r"[a-fA-F0-9]{40}", source.get("revision", "")):
        raise ValueError("Pin source revision to a 40-character repository commit")
    if source.get("dataset") != "uonlp/CulturaX" or source.get("language") != "en":
        raise ValueError("This pilot requires a mapped public CulturaX English release")
    if not source.get("selection_rule") or not source.get("shards"):
        raise ValueError("Declare shard selection rule and inventory")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing preparation output: {output}")
    paths = set()
    for shard in source["shards"]:
        path = (source_manifest.parent / shard["path"]).resolve()
        if path in paths:
            raise ValueError(f"Repeated source shard {path}")
        paths.add(path)
        if sha256(path) != shard["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        print(json.dumps(dict(source_verified=str(path))), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "source_manifest.json", source)
    stats = {s: dict(documents=0, tokens=0) for s in ("train", "validation", "test")}
    scanned = duplicate = excluded = 0
    print(json.dumps(dict(preparation_started=True, output=str(output), seed=seed,
                          sample_fraction=sample_fraction)), flush=True)
    database = sqlite3.connect(output / "documents.sqlite")
    database.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, content_sha256 TEXT UNIQUE, "
                     "split_key TEXT, split TEXT, shard TEXT, row_number INTEGER, "
                     "source TEXT, url TEXT, timestamp TEXT, token_offset INTEGER, tokens INTEGER)")
    try:
        with ExitStack() as stack:
            handles = {s: stack.enter_context((output / f"{s}.bin").open("xb")) for s in stats}
            for shard in source["shards"]:
                path = (source_manifest.parent / shard["path"]).resolve()
                for row_number, row in enumerate(rows_from_shard(path)):
                    scanned += 1
                    if not isinstance(row.get("text"), str) or not row["text"].strip():
                        excluded += 1
                        continue
                    stable_id, content_hash, split_key = document_keys(row, cluster_field)
                    if hash_fraction(split_key, f"sample:{seed}") >= sample_fraction:
                        excluded += 1
                        continue
                    value = hash_fraction(split_key, f"split:{seed}")
                    split = ("validation" if value < validation_fraction else "test"
                             if value < validation_fraction + test_fraction else "train")
                    # Insert before tokenization so duplicate documents are not re-tokenized.
                    cursor = database.execute("INSERT OR IGNORE INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (stable_id, content_hash, split_key, split, shard["path"], row_number,
                         str(row.get("source") or ""), str(row.get("url") or ""),
                         str(row.get("timestamp") or ""), stats[split]["tokens"], 0))
                    if cursor.rowcount == 0:
                        if cluster_field:
                            previous = database.execute("SELECT split_key FROM documents WHERE content_sha256=?",
                                                        (content_hash,)).fetchone()
                            if previous is not None and previous[0] != split_key:
                                raise ValueError("Identical text has conflicting duplicate-cluster IDs")
                        duplicate += 1
                        continue
                    ids = tokenizer.encode(row["text"], add_special_tokens=False, truncation=False)
                    ids.append(tokenizer.eos_token_id)
                    if min(ids) < 0 or max(ids) >= len(tokenizer):
                        raise ValueError("Tokenizer produced IDs outside its declared vocabulary")
                    handles[split].write(np.asarray(ids, dtype="<u4").tobytes())
                    database.execute("UPDATE documents SET tokens=? WHERE id=?", (len(ids), stable_id))
                    stats[split]["documents"] += 1
                    stats[split]["tokens"] += len(ids)
                    if scanned % 10000 == 0:
                        database.commit()
                database.commit()
                print(json.dumps(dict(shard=shard["path"], scanned=scanned, splits=stats)), flush=True)
    finally:
        database.close()
    for split, values in stats.items():
        values["blocks"] = values["tokens"] // sequence_length
        values["packed_tokens"] = values["blocks"] * sequence_length
        values["scored_targets"] = values["blocks"] * (sequence_length - 1)
        values["unused_tail_tokens"] = values["tokens"] % sequence_length
        values["sha256"] = sha256(output / f"{split}.bin")
        if values["packed_tokens"] < (minimum_tokens or {}).get(split, 0):
            raise ValueError(f"{split} has insufficient packed tokens; no completion manifest written")
    report = dict(format_version=1, complete=True, sequence_length=sequence_length,
                  vocab_size=len(tokenizer), eos_token_id=tokenizer.eos_token_id,
                  seed=seed, sample_fraction=sample_fraction, validation_fraction=validation_fraction,
                  test_fraction=test_fraction, cluster_field=cluster_field,
                  split_rule="SHA256(split:seed:cluster-or-text-digest), first 64 bits / 2**64",
                  stable_id_rule="SHA256(JSON([source,url,timestamp,SHA256(text)]))",
                  packing_order="source manifest shard order, then source row order; no document truncation",
                  masking="ordinary causal attention across EOS; no padding; HF labels shift by one",
                  exact_duplicates_removed=duplicate, near_duplicate_audit="not_performed",
                  scanned_documents=scanned, excluded_documents=excluded, splits=stats,
                  tokenizer=tokenizer_provenance or {},
                  preprocessing_sha256=sha256(__file__),
                  software={name: importlib.metadata.version(name) for name in
                            ("numpy", "pyarrow", "transformers", "tokenizers")},
                  source_manifest_sha256=sha256(output / "source_manifest.json"),
                  documents_sha256=sha256(output / "documents.sqlite"))
    write_json(output / "manifest.json", report)  # Completion artifact written last.
    return report


class PackedTokens(Dataset):
    def __init__(self, directory, split, *, verify_hash=False):
        self.directory, self.split = Path(directory), split
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown split")
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if not self.manifest.get("complete") or self.manifest["format_version"] != 1:
            raise ValueError("Incomplete or unsupported token artifact")
        self.length = self.manifest["sequence_length"]
        self.info = self.manifest["splits"][split]
        self.path = self.directory / f"{split}.bin"
        if self.path.stat().st_size != self.info["tokens"] * 4:
            raise ValueError(f"Truncated/oversized token file {self.path}")
        if verify_hash and sha256(self.path) != self.info["sha256"]:
            raise ValueError(f"Token checksum mismatch {self.path}")
        if self.info["blocks"] <= 0:
            raise ValueError(f"No full packs in {split}")
        self._tokens = None

    def __len__(self):
        return self.info["blocks"]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype="<u4", mode="r")
        x = torch.from_numpy(self._tokens[index*self.length:(index+1)*self.length].astype(np.int64))
        return dict(input_ids=x, labels=x.clone())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-fraction", type=float, default=1.)
    parser.add_argument("--validation-fraction", type=float, default=.004)
    parser.add_argument("--test-fraction", type=float, default=.004)
    parser.add_argument("--cluster-field")
    parser.add_argument("--min-train-tokens", type=int, default=5_000_000_000)
    parser.add_argument("--min-validation-tokens", type=int, default=20_000_000)
    parser.add_argument("--min-test-tokens", type=int, default=20_000_000)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-fA-F0-9]{40}", args.tokenizer_revision):
        parser.error("Tokenizer revision must be a pinned commit SHA")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    if len(tokenizer) != 50257 or tokenizer.eos_token_id != 50256:
        parser.error("Use the unchanged GPT-2 tokenizer: V=50257, EOS=50256")
    # Do not let GPT-2's historical 1024-position warning imply document truncation.
    tokenizer.model_max_length = 10**30
    files = {str(p.relative_to(args.tokenizer_dir)): sha256(p)
             for p in sorted(Path(args.tokenizer_dir).rglob("*")) if p.is_file()}
    report = prepare(args.source_manifest, tokenizer, args.output, seed=args.seed,
                     sample_fraction=args.sample_fraction, validation_fraction=args.validation_fraction,
                     test_fraction=args.test_fraction, cluster_field=args.cluster_field,
                     tokenizer_provenance=dict(repo="openai-community/gpt2", revision=args.tokenizer_revision,
                                               files=files),
                     minimum_tokens=dict(train=args.min_train_tokens,
                                         validation=args.min_validation_tokens, test=args.min_test_tokens))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
