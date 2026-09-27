"""Prepare one fixed, disk-backed English token stream for every arm."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from pcc.model import Context
from pcc.packing import document_end_id, preprocessing_policy, tokenize_documents


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_batches(directory, size=1000):
    from datasets import load_from_disk
    directory = Path(directory)
    shards = [directory] if (directory / "state.json").is_file() else sorted(directory.glob("shard_*"))
    if not shards:
        raise ValueError(f"No sampled Arrow datasets: {directory}")
    pending = []
    for path in shards:
        dataset = load_from_disk(str(path))
        if dataset.column_names != ["text"]:
            raise ValueError(f"Expected text-only sampled data: {path}")
        for batch in dataset.iter(batch_size=size):
            pending.extend(batch["text"])
            while len(pending) >= size:
                yield pending[:size]
                pending = pending[size:]
    if pending:
        yield pending


def write_split(directory, split, batches, tokenizer, rows, context, seed):
    end_id = document_end_id(tokenizer)
    path = Path(directory) / f"{split}.bin"
    written = documents = 0
    with path.open("xb") as handle:
        for texts in batches:
            documents += len(texts)
            encoded = tokenize_documents(texts, tokenizer, end_id)
            flat = np.fromiter((x for row in encoded for x in row), dtype=np.uint32)
            count = min(len(flat) // context, rows - written)
            handle.write(flat[:count * context].tobytes())
            written += count
            if written == rows:
                break
        if written != rows:
            raise ValueError(f"Insufficient {split} data: {written} contexts, need {rows}")
    order = np.random.default_rng(seed).permutation(rows).astype(np.int64)
    order_path = Path(directory) / f"{split}.order.npy"
    with order_path.open("xb") as handle:
        np.save(handle, order, allow_pickle=False)
    return {"rows": rows, "context": context, "dtype": "uint32", "documents_tokenized": documents,
            "tokens_sha256": sha256(path), "order_sha256": sha256(order_path),
            "preprocessing": preprocessing_policy(context, end_id)}


def prepare(config, output, recipe):
    import transformers
    from transformers import AutoTokenizer
    if transformers.__version__ != "5.9.0":
        raise ValueError("Prepare this pilot with transformers==5.9.0")
    recipe.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True)
    manifest = {"format": "deep-kv-data-v2", "recipe": asdict(recipe), "sources": config,
                "transformers_version": transformers.__version__,
                "tokenizer_files": {p.name: sha256(p) for p in Path(config["tokenizer"]).iterdir()
                                    if p.is_file() and p.name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")},
                "model_config_sha256": sha256(config["model_config"]),
                "packing": "full_causal", "selection": "packed source prefix, then fixed context permutation"}
    for split, rows in (("train", recipe.train_rows), ("eval", recipe.eval_rows)):
        manifest[split] = write_split(output, split, text_batches(config[f"{split}_data"]),
                                     tokenizer, rows, recipe.context, recipe.data_seed)
        print(f"Prepared {split}: {rows} contexts / {rows * recipe.context} input tokens", flush=True)
    (output / "complete.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class TokenStream:
    def __init__(self, directory, split, recipe, *, verify=True):
        directory = Path(directory)
        self.manifest = json.loads((directory / "complete.json").read_text())
        if self.manifest.get("format") != "deep-kv-data-v2" or self.manifest["recipe"] != asdict(recipe):
            raise ValueError("Prepared stream recipe mismatch")
        spec = self.manifest[split]
        rows = recipe.train_rows if split == "train" else recipe.eval_rows
        if (spec["rows"], spec["context"], spec["dtype"]) != (rows, recipe.context, "uint32"):
            raise ValueError("Prepared stream shape mismatch")
        if spec["preprocessing"] != preprocessing_policy(recipe.context, spec["preprocessing"]["document_end_token_id"]):
            raise ValueError("Wrong document packing policy")
        path, order = directory / f"{split}.bin", directory / f"{split}.order.npy"
        if path.stat().st_size != rows * recipe.context * 4:
            raise ValueError("Token file length mismatch")
        if verify and (sha256(path) != spec["tokens_sha256"] or sha256(order) != spec["order_sha256"]):
            raise ValueError("Prepared stream checksum mismatch")
        self.ids = np.memmap(path, mode="r", dtype=np.uint32, shape=(rows, recipe.context))
        self.order = np.load(order, allow_pickle=False, mmap_mode="r")
        if self.order.shape != (rows,) or self.order.dtype != np.int64:
            raise ValueError("Invalid context permutation")
        if verify and not np.array_equal(np.sort(self.order), np.arange(rows)):
            raise ValueError("Order must visit each context exactly once")

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        # Trainer's collator stacks full contexts; the stored permutation is
        # already seeded, so the Trainer uses a sequential sampler over it.
        return {"input_ids": torch.from_numpy(self.ids[self.order[index]].astype(np.int64))}

    def batch(self, start, stop, device):
        if not 0 <= start < stop <= len(self):
            raise ValueError("Invalid stream slice")
        ids = torch.tensor(self.ids[self.order[start:stop]].astype(np.int64), device=device)
        return Context(ids, torch.ones_like(ids, dtype=torch.bool),
                       torch.arange(ids.shape[1], device=device).expand_as(ids))
