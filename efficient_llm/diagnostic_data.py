"""Freeze a small, disjoint single-document probe from local CulturaX shards."""
import hashlib
import json
from pathlib import Path
import random

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_documents(dataset, tokenizer, *, count, length, seed, excluded,
                     source, eod_id):
    """One fixed prefix per unique sufficiently long document; no cross-doc packing."""
    if count < 1 or length < 1:
        raise ValueError("Document count and sequence length must be positive")
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    rows, records = [], []
    for index in order:
        text = dataset[index]["text"]
        identity = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if identity in excluded:
            continue
        tokens = tokenizer(text, add_special_tokens=False, truncation=True,
                           max_length=length + 1)["input_ids"]
        # A real document end can complete a window. A truncated prefix never
        # receives an artificial EOS inside the scored window.
        if not tokens or tokens[-1] != eod_id:
            tokens.append(eod_id)
        if len(tokens) < length + 1:
            continue
        prefix = tokens[:length + 1]
        token_hash = hashlib.sha256(json.dumps(prefix, separators=(",", ":")).encode()).hexdigest()
        if token_hash in excluded:
            continue
        excluded.update((identity, token_hash))
        rows.append(prefix)
        records.append({"source": source, "row": index, "text_sha256": identity,
                        "tokens_sha256": token_hash})
        if len(rows) == count:
            break
    return rows, records


def prepare_probe(data_root, model_root, output, *, length=2048, calibration=64,
                  evaluation=128, seed=42, max_shards=8, contract_path):
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from training_data import document_end
    data_root, model_root, output = map(Path, (data_root, model_root, output))
    if output.exists() or min(length, calibration, evaluation, max_shards) < 1:
        raise ValueError("Require a fresh probe directory and positive sizes")
    contract = json.loads(Path(contract_path).read_text())
    if not contract.get("model_weights"):
        raise ValueError("Pretrained weight identity is required")
    for name, expected in contract["model_weights"].items():
        weight = model_root / name
        if weight.stat().st_size != expected["bytes"] or sha256(weight) != expected["sha256"]:
            raise ValueError("Pretrained weight checksum mismatch: " + name)
    subset = data_root / "subsets" / contract["subset"]
    if sha256(subset / "manifest.json") != contract["manifest_sha256"]:
        raise ValueError("Pinned pool manifest differs")
    if sha256(data_root / "UPLOAD_COMPLETE.json") != contract["completion_sha256"]:
        raise ValueError("Pinned pool completion record differs")
    manifest = json.loads((subset / "manifest.json").read_text())
    for name, expected in contract["tokenizer_files"].items():
        if sha256(model_root / name) != expected["sha256"]:
            raise ValueError("Model config/tokenizer differs: " + name)
    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    eod = document_end(tokenizer, "<|endoftext|>")
    excluded, tensors, selections, verified = set(), {}, {}, {}
    for split, name, count in (("train", "calibration", calibration),
                               ("validation", "evaluation", evaluation)):
        rows, records = [], []
        # Bound I/O: deterministic rows in the first pinned shards. This is not
        # claimed to be a uniform sample of the full 30B-token training pool.
        for shard_index, shard in enumerate(manifest["shards"][split][:max_shards]):
            for relative, expected in shard["files"].items():
                path = (data_root / relative).resolve()
                if not path.is_relative_to(data_root.resolve()):
                    raise ValueError("Manifest path escapes dataset root")
                if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
                    raise ValueError("Shard checksum mismatch: " + relative)
                verified[relative] = expected
            # Manifest shard path may be relative to the repository root or the
            # subset. File records identify the actual saved-dataset directory.
            states = [data_root / p for p in shard["files"] if Path(p).name == "state.json"]
            if len(states) != 1:
                raise ValueError("Expected one saved dataset per manifest shard")
            ds = load_from_disk(str(states[0].parent))
            selected, ids = choose_documents(ds, tokenizer, count=count-len(rows),
                length=length, seed=seed + (10000 if split == "validation" else 0) + shard_index,
                excluded=excluded, source=str(states[0].parent.relative_to(data_root)), eod_id=eod["id"])
            rows.extend(selected); records.extend(ids)
            if len(rows) == count:
                break
        if len(rows) != count:
            raise ValueError(f"Only {len(rows)}/{count} eligible {name} documents in bounded source pool")
        tensors[name] = torch.tensor(rows, dtype=torch.long)
        selections[name] = records
    output.mkdir(parents=True)
    torch.save(tensors, output / "tokens.pt")
    model_files = {p.name: sha256(p) for p in sorted(model_root.iterdir())
                   if p.is_file() and p.suffix in {".json", ".safetensors"}}
    receipt = {"status": "passed", "length": length, "seed": seed,
        "calibration_documents": calibration, "evaluation_documents": evaluation,
        "selection": "seeded document prefixes in bounded pinned shards; long documents only",
        "eod": eod, "data_root": str(data_root), "model_root": str(model_root),
        "pool_revision": contract["revision"], "tokenizer_revision": contract["tokenizer_revision"],
        "tokens_sha256": sha256(output / "tokens.pt"), "model_files": model_files,
        "verified_shard_files": verified, "documents": selections}
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
