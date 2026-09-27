"""Load sampled text and reuse train.py's Hugging Face cached CLM pipeline."""
import hashlib
from pathlib import Path

from pcc.packing import document_end_id, preprocess_dataset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_text(directory):
    from datasets import concatenate_datasets, load_from_disk
    directory = Path(directory)
    paths = [directory] if (directory / "state.json").is_file() else sorted(directory.glob("shard_*"))
    if not paths:
        raise ValueError(f"No sampled Arrow datasets: {directory}")
    datasets = [load_from_disk(str(path)) for path in paths]
    return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]


def load_data(config, recipe, training_args):
    from transformers import AutoTokenizer
    recipe.validate()
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True)
    if recipe.context > tokenizer.model_max_length:
        raise ValueError("Context exceeds tokenizer model_max_length")
    workers = config.get("preprocessing_num_workers", 1)
    overwrite = config.get("overwrite_cache", False)
    datasets = {}
    metadata = {"format": "deep-kv-hf-data-v1", "context": recipe.context,
                "map_batch_size": 1000, "document_end_token_id": document_end_id(tokenizer),
                "preprocessing_num_workers": {"train": workers, "eval": 1},
                "shuffle_seed": training_args.seed, "splits": {}}
    for split in ("train", "eval"):
        raw = load_text(config[f"{split}_data"])
        # The small eval split was audited with one worker. Multiprocess map
        # drops a remainder at each worker boundary and could exhaust its margin.
        packed = preprocess_dataset(raw, tokenizer, recipe.context, training_args,
                                    num_proc=workers if split == "train" else 1,
                                    overwrite_cache=overwrite)
        needed = recipe.train_rows if split == "train" else recipe.eval_rows
        if len(packed) < needed:
            raise ValueError(f"Insufficient {split} data: {len(packed)} contexts, need {needed}")
        # Match train.py: shuffle the dataset once, then use Trainer's normal
        # seeded training sampler. Do not trim the training source to a prefix.
        data = packed.shuffle(seed=training_args.seed) if split == "train" else packed.select(range(needed))
        metadata["splits"][split] = {"source": str(Path(config[f"{split}_data"]).resolve()),
                                    "source_fingerprint": raw._fingerprint,
                                    "dataset_fingerprint": data._fingerprint, "rows": len(data)}
        datasets[split] = data
    return datasets["train"], datasets["eval"], metadata
