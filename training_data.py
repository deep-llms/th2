"""Local text loading and cached HF CLM packing shared by training arms.

Track source-document IDs through packing for isolated causal attention,
position resets and target masking. Drop remainders per map batch/worker.
"""
import hashlib
import json
from itertools import chain
from pathlib import Path
import re

import datasets
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
import transformers
import tokenizers
import pyarrow
import pyarrow.compute as pc
import numpy as np

POLICY = "document_segments_map_v2"
SPLITS = {"train", "training", "validation", "valid", "val", "dev", "eval", "test"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_record(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def discover_sources(path, ancestors=()):
    """Sorted leaves only. Never recursively combine named train/eval splits."""
    path = Path(path).resolve()
    if path in ancestors:
        raise ValueError(f"Dataset directory cycle: {path}")
    if path.is_file() and path.suffix == ".parquet":
        return [("parquet", path)]
    if not path.is_dir():
        raise ValueError(f"Expected a local Parquet file or dataset directory: {path}")
    if (path / "dataset_dict.json").exists():
        raise ValueError(f"Select one split, not a saved DatasetDict: {path}")
    if (path / "state.json").is_file():
        return [("saved", path)]
    children = sorted(p for p in path.iterdir() if not p.name.startswith("."))
    dirs = [p for p in children if p.is_dir()]
    files = [p for p in children if p.is_file() and p.suffix == ".parquet"]
    if any(p.name.lower() in SPLITS for p in dirs):
        raise ValueError(f"Select a single split directory instead of its parent: {path}")
    split_names = {m.group(1).lower() for p in files
                   if (m := re.match(r"(train|training|validation|valid|val|dev|eval|test)[_.-]", p.name, re.I))}
    if len(split_names) > 1:
        raise ValueError(f"Parquet directory contains multiple named splits: {path}")
    if files and dirs:
        raise ValueError(f"Ambiguous mixed Parquet/directory layout: {path}")
    if files:
        return [("parquet", p.resolve()) for p in files]
    if not dirs:
        raise ValueError(f"No saved dataset or Parquet files in {path}")
    return [source for child in dirs for source in discover_sources(child, (*ancestors, path))]


def load_text(path, cache_dir=None, keep_document_ids=False):
    """Load one explicitly selected split; discard non-text metadata columns."""
    sources = discover_sources(path)
    paths = [str(p) for _, p in sources]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate source paths (including symlink aliases)")
    pieces, records = [], []
    # Load contiguous Parquet groups together, keeping declared lexical order.
    groups = []
    for kind, source in sources:
        if kind == "parquet" and groups and groups[-1][0] == kind:
            groups[-1][1].append(source)
        else:
            groups.append((kind, [source]))
    for kind, group in groups:
        if kind == "saved":
            ds = load_from_disk(str(group[0]))
            files = [group[0] / "state.json", group[0] / "dataset_info.json"]
            files += [Path(f["filename"]) for f in ds.cache_files]
        else:
            ds = load_dataset("parquet", data_files={"train": [str(p) for p in group]},
                              split="train", cache_dir=cache_dir, columns=["text", "id"] if keep_document_ids else ["text"])
            files = group
        if not isinstance(ds, Dataset) or "text" not in ds.column_names:
            raise ValueError(f"Expected a Dataset with a text column: {group}")
        if getattr(ds.features["text"], "dtype", None) not in ("string", "large_string"):
            raise ValueError(f"text must be a string column: {group}")
        records.append({"format": kind, "sources": [str(p) for p in group],
                        "files": [file_record(p) for p in files],
                        "rows": len(ds), "fingerprint": ds._fingerprint})
        pieces.append(ds.select_columns(["text", "id"] if keep_document_ids else ["text"]))
    raw = pieces[0] if len(pieces) == 1 else concatenate_datasets(pieces)
    if not len(raw):
        raise ValueError("Selected training split is empty")
    return raw, records


def document_end(tokenizer, requested=None):
    token = requested if requested is not None else tokenizer.eos_token
    if not token:
        raise ValueError("Tokenizer has no EOS; set --document_end_token explicitly")
    token_id = tokenizer.get_vocab().get(token)
    if token_id is None or tokenizer.encode(token, add_special_tokens=False) != [token_id]:
        raise ValueError("document_end_token must be an existing single vocabulary token")
    return {"token": token, "id": token_id, "selection": "explicit" if requested is not None else "tokenizer_eos"}


def tokenizer_identity(tokenizer):
    """Content identity, independent of local tokenizer path or output directory."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    return {"class": type(tokenizer).__name__,
            "backend_sha256": digest(json.loads(backend.to_str())) if backend is not None else None,
            "vocab_sha256": digest(tokenizer.get_vocab()),
            "special_tokens": {k: str(v) for k, v in tokenizer.special_tokens_map.items()},
            "model_max_length": tokenizer.model_max_length,
            "padding_side": tokenizer.padding_side, "truncation_side": tokenizer.truncation_side}


def tokenizer_files(directory):
    """Small tokenizer assets only; never hash/copy model weights or credentials."""
    names = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
             "vocab.json", "vocab.txt", "merges.txt", "tokenizer.model", "spiece.model",
             "sentencepiece.bpe.model"}
    return [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(Path(directory).iterdir()) if p.is_file() and (p.name in names or p.suffix == ".py")]


def tokenize_batch(examples, tokenizer, end_id):
    if any(not isinstance(text, str) for text in examples["text"]):
        raise ValueError("Every document must be a string; null text is not silently skipped")
    ids = tokenizer(examples["text"], add_special_tokens=False,
                    truncation=False, padding=False, return_attention_mask=False,
                    return_token_type_ids=False)["input_ids"]
    ids = [list(row) + [end_id] for row in ids]
    return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}


def tokenize_with_segments(examples, indices, tokenizer, end_id):
    result = tokenize_batch(examples, tokenizer, end_id)
    result["segments"] = [[index] * len(ids) for index, ids in zip(indices, result["input_ids"])]
    return result


def group_texts(examples, block_size, isolate_documents=True):
    joined = {key: list(chain.from_iterable(rows)) for key, rows in examples.items()}
    length = len(joined["input_ids"]) // block_size * block_size
    result = {key: [row[i:i + block_size] for i in range(0, length, block_size)]
              for key, row in joined.items()}
    if not result["input_ids"]:
        result["labels"], result["target_count"] = [], []
        return result
    # Vectorize boundary masking/counting: avoid a Python operation per token
    # across a multi-billion-token corpus. Values remain ordinary unshifted IDs.
    labels = np.asarray(result["input_ids"], dtype=np.int64).copy()
    labels[:, 0] = -100
    if isolate_documents:
        segments = np.asarray(result["segments"], dtype=np.int64)
        labels[:, 1:][segments[:, 1:] != segments[:, :-1]] = -100
    result["labels"] = labels.tolist()
    result["target_count"] = np.count_nonzero(labels != -100, axis=1).tolist()
    return result


def prepare_training_data(data_args, training_args, tokenizer, tokenizer_dir, block_size, attention_backend="sdpa"):
    if attention_backend not in ("sdpa", "fa4"):
        raise ValueError("attention_backend must be sdpa or fa4")
    if block_size < 2 or data_args.map_batch_size < 1:
        raise ValueError("block_size must be >= 2 and map_batch_size must be positive")
    if data_args.reset_document_positions and not data_args.isolate_documents:
        raise ValueError("Position reset requires document isolation; disable both for the cross-document control")
    workers = data_args.preprocessing_num_workers or 1
    if workers < 1 or data_args.preprocessing_num_workers == 0:
        raise ValueError("preprocessing_num_workers must be positive")
    stream = getattr(data_args, "data_order", "packed_shuffle") == "document_stream"
    split_receipt = None
    with training_args.main_process_first(desc="load local text"):
        raw, sources = load_text(data_args.data_dir, data_args.data_cache_dir, keep_document_ids=stream)
        if stream:
            from bdot.data import document_split
            raw, split_receipt = document_split(raw, data_args, tokenizer)
    effective_workers = min(workers, len(raw))
    end = document_end(tokenizer, data_args.document_end_token)
    seed = training_args.data_seed if training_args.data_seed is not None else training_args.seed
    recipe = {"policy": POLICY, "sources": sources, "raw_fingerprint": raw._fingerprint,
              "raw_documents": len(raw), "tokenizer": tokenizer_identity(tokenizer),
              "tokenizer_files": tokenizer_files(tokenizer_dir), "document_end": end,
              "block_size": block_size, "map_batch_size": data_args.map_batch_size,
              "requested_workers": workers, "effective_workers": effective_workers,
              "shuffle_seed": seed, "add_special_tokens": False,
              "append_boundary": "one_after_every_document_including_empty",
              "remainder": "drop_per_map_batch_within_worker_shard",
              "attention": "causal_within_document" if data_args.isolate_documents else "causal_across_documents",
              "positions": "reset_per_document_fragment" if data_args.reset_document_positions else "continuous_within_block",
              "labels": "ignore_first_and_document_starts" if data_args.isolate_documents else "ignore_block_start",
              "attention_backend": attention_backend,
              "boundary_code_sha256": hashlib.sha256(Path(__file__).with_name("packed_training.py").read_bytes()).hexdigest(),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "versions": {"datasets": datasets.__version__, "transformers": transformers.__version__,
                           "tokenizers": tokenizers.__version__, "pyarrow": pyarrow.__version__}}
    if stream:
        recipe.update(policy="document_stream_v1", shuffle_seed=0,
                      document_manifest=split_receipt, remainder="drop_final_stream_tail_only",
                      stream_code_sha256=hashlib.sha256(Path(__file__).with_name("bdot").joinpath("data.py").read_bytes()).hexdigest())
    # Include worker count explicitly: grouping boundaries can change with num_proc.
    # Cache files remain ordinary HF dataset caches shared by matching runs.
    token_recipe = {k: v for k, v in recipe.items() if k not in ("shuffle_seed", "block_size", "attention_backend")}
    token_fingerprint = digest({"stage": "tokenize", "recipe": token_recipe})[:32]
    map_kwargs = dict(batched=True, batch_size=data_args.map_batch_size,
                      num_proc=effective_workers if effective_workers > 1 else None,
                      load_from_cache_file=not data_args.overwrite_cache)
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized = raw.map(tokenize_with_segments, with_indices=True, fn_kwargs={"tokenizer": tokenizer, "end_id": end["id"]},
                            remove_columns=raw.column_names, new_fingerprint=token_fingerprint,
                            desc="Tokenize with document EOS", **map_kwargs)
    with training_args.main_process_first(desc="dataset map packing"):
        if stream:
            from bdot.data import pack_document_stream
            packed = pack_document_stream(tokenized, data_args, block_size)
        else:
            packed = tokenized.map(group_texts, fn_kwargs={"block_size": block_size, "isolate_documents": data_args.isolate_documents},
                                   new_fingerprint=digest({"stage": "pack", "tokenized": token_fingerprint, "block_size": block_size})[:32],
                                   desc="Pack fixed-length sequences", **map_kwargs)
    if not len(packed):
        raise ValueError("No complete training sequences after packing; supply more text or a smaller block_size")
    target_count = pc.sum(packed.data.column("target_count")).as_py()
    if not target_count:
        raise ValueError("Packed dataset has no within-document prediction targets")
    packed = packed.remove_columns("target_count")
    with training_args.main_process_first(desc="shuffle packed sequences"):
        shuffled = packed if stream else packed.shuffle(seed=seed, load_from_cache_file=not data_args.overwrite_cache)
    recipe.update(tokenized_fingerprint=tokenized._fingerprint, packed_fingerprint=packed._fingerprint,
                  shuffled_fingerprint=shuffled._fingerprint, packed_sequences=len(packed),
                  packed_input_tokens=len(packed) * block_size,
                  scored_target_tokens=target_count)
    if attention_backend == "fa4":
        from importlib.metadata import version
        recipe["fa4_runtime"] = {
            "package": "flash-attn-4", "version": version("flash-attn-4"),
            "adapter_sha256": hashlib.sha256(Path(__file__).with_name("packed_fa4.py").read_bytes()).hexdigest()}
    return shuffled, recipe


def require_resume_data(checkpoint, recipe):
    path = Path(checkpoint) / "data_recipe.json"
    if not path.is_file():
        raise ValueError("Resume checkpoint lacks data_recipe.json; cannot verify EOS/data compatibility. Use a fresh run.")
    if json.loads(path.read_text()) != recipe:
        raise ValueError("Resume data/tokenizer/packing differs from checkpoint; use the original recipe or a fresh run")


def save_data_recipe(directory, recipe):
    with (Path(directory) / "data_recipe.json").open("w") as handle:
        json.dump(recipe, handle, indent=2, ensure_ascii=False, allow_nan=False)


def require_held_out(train_recipe, eval_recipe):
    """Reject shared resolved source paths; this is not content deduplication."""
    a, b = train_recipe.get("document_manifest"), eval_recipe.get("document_manifest")
    if a and b and a["sha256"] == b["sha256"] and a["split"] == "train" and b["split"] == "validation":
        return  # Same physical corpus, explicit disjoint document-ID selection.
    def paths(recipe):
        return {str(Path(p).resolve()) for source in recipe["sources"]
                for p in [*source["sources"], *(f["path"] for f in source["files"])]}
    overlap = paths(train_recipe) & paths(eval_recipe)
    if overlap:
        raise ValueError(f"Validation overlaps training source paths: {sorted(overlap)[:3]}")
