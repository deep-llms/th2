"""Count token frequencies over the training corpus.

Mirrors train.py's data loading (per-language dirs, shard-aware) and
tokenization (same tokenizer call, add_special_tokens=False), on a
deterministic 1-in-N document sample per language. Document sampling is
uniform, so relative token frequencies (all that binning needs) are unbiased;
absolute counts scale by ~N.

Saves an .npz with:
    counts          int64[vocab]  total counts across all languages (sampled)
    counts_<lang>   int64[vocab]  per-language counts (sampled)
plus a JSON sidecar with sample settings and totals.

Usage:
  python scripts/count_token_freq.py \
      --data-dir /opt/dlami/nvme/sparse_emb_data/Qwen_Qwen3-0.6B/train \
      --tokenizer-name Qwen/Qwen3-0.6B \
      --sample-every 10 --num-workers 64 \
      --output /opt/dlami/nvme/sparse_emb_outputs/token_freq.npz
"""

import argparse
import json
import os

import numpy as np
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer


def load_lang_dataset(lang_path):
    """Load one language dir, concatenating shard_* subdirs if present
    (same layout handling as train.py)."""
    shard_dirs = sorted(
        os.path.join(lang_path, d) for d in os.listdir(lang_path)
        if d.startswith("shard_") and os.path.isdir(os.path.join(lang_path, d)))
    if shard_dirs:
        return concatenate_datasets([load_from_disk(sd) for sd in shard_dirs])
    return load_from_disk(lang_path)


def main():
    parser = argparse.ArgumentParser(description="Token frequency counts")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--vocab-size", type=int, default=151936,
                        help="Array size; must be >= max token id (Qwen3: 151936)")
    parser.add_argument("--sample-every", type=int, default=10,
                        help="Keep every Nth document (1 = full corpus)")
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--batch-rows", type=int, default=10000,
                        help="Documents per counting batch")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    def tokenize_function(examples):
        return {"input_ids": tokenizer(examples["text"],
                                       add_special_tokens=False)["input_ids"]}

    arrays = {}
    total = np.zeros(args.vocab_size, dtype=np.int64)
    meta = {"sample_every": args.sample_every, "langs": {}}

    lang_dirs = sorted(
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d)))
    if not lang_dirs:
        raise ValueError(f"No language dirs in {args.data_dir}")

    for lang in lang_dirs:
        ds = load_lang_dataset(os.path.join(args.data_dir, lang))
        n_docs = ds.num_rows
        sampled = ds.select(range(0, n_docs, args.sample_every))
        print(f"[{lang}] {n_docs:,} docs -> sampling {sampled.num_rows:,}")

        tokenized = sampled.map(
            tokenize_function,
            batched=True,
            num_proc=args.num_workers,
            remove_columns=sampled.column_names,
            desc=f"Tokenizing {lang}",
        )

        counts = np.zeros(args.vocab_size, dtype=np.int64)
        tokenized = tokenized.with_format("numpy")
        for start in range(0, tokenized.num_rows, args.batch_rows):
            batch = tokenized[start:start + args.batch_rows]["input_ids"]
            # dtype forced: empty docs tokenize to [], whose default float64
            # would promote the concatenation and break bincount
            flat = np.concatenate(
                [np.asarray(row, dtype=np.int64) for row in batch])
            if flat.size == 0:
                continue
            if flat.max() >= args.vocab_size:
                raise ValueError(
                    f"token id {flat.max()} >= vocab_size {args.vocab_size}")
            counts += np.bincount(flat, minlength=args.vocab_size)

        arrays[f"counts_{lang}"] = counts
        total += counts
        meta["langs"][lang] = {
            "docs_total": int(n_docs),
            "docs_sampled": int(sampled.num_rows),
            "tokens_counted": int(counts.sum()),
        }
        print(f"[{lang}] tokens counted: {counts.sum():,}  "
              f"unique ids: {int((counts > 0).sum()):,}")

    arrays["counts"] = total
    np.savez_compressed(args.output, **arrays)
    with open(args.output.replace(".npz", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nTotal tokens counted: {total.sum():,}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
