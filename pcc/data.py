"""Shared deterministic loading of fixed sampler splits; no export stage."""
from itertools import chain
import json
from pathlib import Path

import numpy as np
import torch

from .model import Context
from .protocol import CONTEXT, DATA_SEED, MODEL_ID, REVISION


def save_contexts(path, input_ids, valid, position_ids, metadata, segments=None):
    arrays = dict(input_ids=np.asarray(input_ids), valid=np.asarray(valid),
                  position_ids=np.asarray(position_ids), metadata=np.array(json.dumps(metadata)))
    if segments is not None:
        arrays["segments"] = np.asarray(segments)
    with Path(path).open("xb") as handle:
        np.savez(handle, **arrays)


class PreparedContexts:
    def __init__(self, path, split, token_budget):
        self.path = Path(path).resolve()
        if self.path.is_dir():
            raise ValueError(
                "Use SampledDataLoader with the model tokenizer for raw-text Arrow datasets."
            )
        with np.load(self.path, allow_pickle=False) as archive:
            fields = set(archive.files)
            required_fields = {"input_ids", "valid", "position_ids", "metadata"}
            if not required_fields <= fields or fields - (required_fields | {"segments"}):
                raise ValueError("Missing or unknown context archive fields")
            self.ids = archive["input_ids"].copy()
            self.valid = archive["valid"].copy()
            self.positions = archive["position_ids"].copy()
            self.segments = archive["segments"].copy() if "segments" in archive else None
            self.metadata = json.loads(str(archive["metadata"].item()))
        self.validate(split, token_budget)

    def validate(self, split, token_budget):
        if not isinstance(self.metadata, dict):
            raise ValueError("Context metadata must be a JSON object")
        shape = self.ids.shape
        if len(shape) != 2 or shape[1] != CONTEXT or shape[0] == 0:
            raise ValueError("Prepared contexts must have nonempty shape [N, 2048]")
        if self.valid.shape != shape or self.positions.shape != shape or self.valid.dtype != np.bool_:
            raise ValueError("Invalid prepared masks/positions")
        if not np.issubdtype(self.ids.dtype, np.integer) or not np.issubdtype(self.positions.dtype, np.integer):
            raise ValueError("Input and position IDs must be integers")
        if (self.ids < 0).any() or (self.positions < 0).any():
            raise ValueError("Input and position IDs must be nonnegative")
        if (self.valid.sum(-1) < 2).any() or np.any(self.valid[:, 1:] & ~self.valid[:, :-1]):
            raise ValueError("Prepared contexts require right padding and at least two tokens per row")
        if int(self.valid.sum()) != token_budget:
            raise ValueError(f"Expected exactly {token_budget} input tokens, got {int(self.valid.sum())}")
        if split == "train" and not self.valid.all():
            raise ValueError("Probe/screen training uses complete 2048-token contexts")
        required = {"split": split, "language": "en", "model_id": MODEL_ID,
                    "tokenizer_revision": REVISION, "data_order_seed": DATA_SEED}
        for key, value in required.items():
            if self.metadata.get(key) != value:
                raise ValueError(f"Prepared metadata {key} must be {value!r}")
        if not self.metadata.get("source") or not self.metadata.get("preprocessing"):
            raise ValueError("Record selected subset/split source and existing preprocessing policy")
        if self.metadata.get("packing") not in ("full_causal", "block_isolated"):
            raise ValueError("Record full_causal or block_isolated packing")
        if (self.segments is not None) != (self.metadata["packing"] == "block_isolated"):
            raise ValueError("Segment IDs must match the declared packing policy")
        if self.segments is not None and (self.segments.shape != shape or not np.issubdtype(self.segments.dtype, np.integer)):
            raise ValueError("Invalid segment IDs")
        eligible = self.valid[:, :-1] & self.valid[:, 1:]
        if self.segments is not None:
            for row, valid in zip(self.segments, self.valid):
                ids = row[valid]
                runs = ids[np.r_[True, ids[1:] != ids[:-1]]]
                if (ids < 0).any() or len(np.unique(runs)) != len(runs):
                    raise ValueError("Isolated segments need distinct nonnegative IDs per contiguous segment")
            eligible &= self.segments[:, :-1] == self.segments[:, 1:]
        if not eligible.any(axis=1).all():
            raise ValueError("Each evaluation/training context must have at least one eligible causal target")

    def __len__(self):
        return len(self.ids)

    def batch(self, start, stop, device):
        if not 0 <= start < stop <= len(self):
            raise ValueError("Invalid batch slice")
        def tensor(array, dtype):
            # Avoid mutable CPU tensor views altering the shared paired stream.
            return torch.tensor(array[start:stop], dtype=dtype, device=device)
        return Context(tensor(self.ids, torch.long), tensor(self.valid, torch.bool),
                       tensor(self.positions, torch.long),
                       tensor(self.segments, torch.long) if self.segments is not None else None)


def validate_matched_data(train, dev):
    if train.path == dev.path or train.metadata["source"] == dev.metadata["source"]:
        raise ValueError("Train and dev must identify distinct fixed split sources")
    if set(train.metadata.get("dataset_shards", ())) & set(dev.metadata.get("dataset_shards", ())):
        raise ValueError("Fixed splits must not share dataset shards")
    for key in ("tokenizer_revision", "model_id", "preprocessing", "packing", "data_order_seed"):
        if train.metadata[key] != dev.metadata[key]:
            raise ValueError(f"Train/dev context policies differ: {key}")


class SampledDataLoader:
    """Tokenize each fixed split once and serve identical shuffled prefixes.

    Matches train.py's single-process, 1000-document map batches: concatenate
    without special tokens, drop each batch's remainder, then shuffle contexts.
    Arrow caches live in the run directory; sampler inputs are never modified.
    NPZ inputs remain supported for existing fixtures and already packed data.
    """

    def __init__(self, tokenizer, cache_dir):
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir)
        self.packed = {}

    def __call__(self, path, split, token_budget):
        path = Path(path).resolve()
        if not path.is_dir():
            return PreparedContexts(path, split, token_budget)
        if type(token_budget) is not int or token_budget < 2 or token_budget % CONTEXT == 1:
            raise ValueError("Invalid input-token budget: each context needs a causal target")
        if split == "train" and token_budget % CONTEXT:
            raise ValueError("Training requires complete contexts")
        if path not in self.packed:
            self.packed[path] = self._pack(path)
        packed, identity = self.packed[path]
        rows = (token_budget + CONTEXT - 1) // CONTEXT
        if rows > len(packed):
            raise ValueError(f"{split}: need {token_budget} input tokens, but the fixed split has "
                             f"{len(packed) * CONTEXT} after packing; no resampling or split borrowing")
        data = PreparedContexts.__new__(PreparedContexts)
        data.path = path
        data.ids = np.asarray(packed[:rows]["input_ids"], dtype=np.int64)
        data.valid = np.arange(rows * CONTEXT).reshape(rows, CONTEXT) < token_budget
        data.ids[~data.valid] = 0
        data.positions = np.broadcast_to(np.arange(CONTEXT, dtype=np.int64), data.ids.shape)
        data.segments = None
        data.metadata = {**identity, "split": split, "language": "en", "model_id": MODEL_ID,
                         "tokenizer_revision": REVISION, "data_order_seed": DATA_SEED,
                         "packing": "full_causal", "preprocessing": {
                             "policy": "legacy_document_map", "map_batch_size": 1000, "num_proc": 1,
                             "context_length": CONTEXT, "add_special_tokens": False,
                             "remainder": "drop_per_document_map_batch"}}
        data.validate(split, token_budget)
        return data

    def _pack(self, path):
        from datasets import Dataset, concatenate_datasets, load_from_disk

        if self.tokenizer is None:
            raise ValueError("Raw-text datasets require the pinned model tokenizer")
        shards = [path] if (path / "dataset_info.json").is_file() else sorted(path.glob("shard_*"))
        if not shards or any(not p.is_dir() for p in shards):
            raise ValueError(f"Expected a saved text Dataset or shard_* directories: {path}")
        shards = [p.resolve() for p in shards]
        if len(set(shards)) != len(shards):
            raise ValueError("Duplicate dataset shard aliases")
        parts = [load_from_disk(str(p)) for p in shards]
        if any(not isinstance(ds, Dataset) or ds.column_names != ["text"] for ds in parts):
            raise ValueError("Sampler inputs must be text-only Dataset splits")
        identity = {"dataset_shards": [str(p) for p in shards],
                    "source": json.dumps([{"path": str(p), "fingerprint": ds._fingerprint, "rows": len(ds)}
                                          for p, ds in zip(shards, parts)], sort_keys=True)}
        raw = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        if not len(raw):
            raise ValueError("Empty sampled split")
        # One fresh cache per split in this run, not a separate prepared dataset.
        cache = self.cache_dir / f"split-{len(self.packed)}"
        cache.mkdir(parents=True, exist_ok=False)
        tokenizer = self.tokenizer
        context_length = CONTEXT

        def tokenize(batch):
            return {"input_ids": tokenizer(batch["text"], add_special_tokens=False)["input_ids"]}

        def pack(batch):
            tokens = list(chain.from_iterable(batch["input_ids"]))
            kept = len(tokens) // context_length * context_length
            return {"input_ids": [tokens[i:i + context_length] for i in range(0, kept, context_length)]}

        tokenized = raw.map(tokenize, batched=True, batch_size=1000, num_proc=None,
                            remove_columns=raw.column_names, cache_file_name=str(cache / "tokenized.arrow"))
        packed = tokenized.map(pack, batched=True, batch_size=1000, num_proc=None,
                               cache_file_name=str(cache / "packed.arrow"))
        packed = packed.shuffle(seed=DATA_SEED, indices_cache_file_name=str(cache / "order.arrow"))
        return packed, identity
