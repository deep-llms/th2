"""Download, tokenize, and sample CulturaX data for cross-lingual training.

Train targets (in tokens):
  - en: 30B tokens
  - vi, zh, ru, de, ar: 300M tokens each

Eval targets (sampled from documents immediately after train set):
  - en, vi, zh, ru, de, ar: 10M tokens each

Pipeline:
  1. List all parquet files per language via HfFileSystem
  2. Randomly select files (seeded for reproducibility)
  3. Download selected files via snapshot_download
  4. Sample documents, using tokenizer to count tokens until target is reached
  5. Continue sampling for eval set (zero overlap with train)
  6. Save raw text as HuggingFace Dataset (Arrow format)

Usage:
  python prepare_data.py --dry-run                    # preview download plan
  python prepare_data.py download                     # download parquet files
  python prepare_data.py sample                       # sample by token count, save raw text
  python prepare_data.py download sample              # both steps
  python prepare_data.py download sample --langs en vi  # specific languages
  python prepare_data.py sample --tokenizer-name meta-llama/Llama-3-8B  # different tokenizer
"""

import argparse
import os
import random

import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import HfFileSystem, snapshot_download
from transformers import AutoTokenizer

REPO_ID = "uonlp/CulturaX"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
SEED = 42

# num_files: how many parquet files to download per language. The previous run
# over-downloaded massively (en hit 30B tokens in 35 of 300 files; every other
# lang hit 300M in its 1st file). At ~2.46GB / ~857M tokens per en file, 50 files
# ≈ 123GB / ~43B tokens — comfortable margin over the 30B+10M target. 5 files per
# other lang ≈ ~1.5B tokens, far above the 300M+10M target. Selection stays
# deterministic via SEED, so it is reproducible across machine restarts (but note:
# this is a NEW selection, not the old run's files — sample(k) is not nested).
LANG_CONFIG = {
    "en": {"target_tokens": 30_000_000_000, "eval_tokens": 10_000_000, "num_files": 50},
    "vi": {"target_tokens": 300_000_000, "eval_tokens": 10_000_000, "num_files": 5},
    "zh": {"target_tokens": 300_000_000, "eval_tokens": 10_000_000, "num_files": 5},
    "ru": {"target_tokens": 300_000_000, "eval_tokens": 10_000_000, "num_files": 5},
    "de": {"target_tokens": 300_000_000, "eval_tokens": 10_000_000, "num_files": 5},
    "ar": {"target_tokens": 300_000_000, "eval_tokens": 10_000_000, "num_files": 5},
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def list_parquet_files(fs, lang):
    path = f"datasets/{REPO_ID}/{lang}"
    files = fs.ls(path, detail=False)
    return sorted([f for f in files if f.endswith(".parquet")])


def select_files(all_files, num_files, rng):
    if num_files is None or num_files >= len(all_files):
        return all_files
    return sorted(rng.sample(all_files, num_files))


def run_download(args, plan):
    for lang, files in plan.items():
        filenames = [os.path.basename(f) for f in files]
        allow_patterns = [f"{lang}/{fn}" for fn in filenames]

        print(f"\n[{lang}] Downloading {len(files)} files...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=args.raw_dir,
            allow_patterns=allow_patterns,
            max_workers=args.num_workers,
        )

        lang_dir = os.path.join(args.raw_dir, lang)
        final_count = len([f for f in os.listdir(lang_dir) if f.endswith(".parquet")])
        print(f"  Done: {final_count} parquet files in {lang_dir}")

    print("\nAll downloads complete.")


# ---------------------------------------------------------------------------
# Sample by token count
# ---------------------------------------------------------------------------

def _flush_shard(texts, shard_dir, shard_idx, label):
    """Save a list of texts as a numbered shard."""
    shard_path = os.path.join(shard_dir, f"shard_{shard_idx:04d}")
    Dataset.from_dict({"text": texts}).save_to_disk(shard_path)
    print(f"    Flushed {label} shard {shard_idx}: {len(texts):,} documents → {shard_path}")


def sample_by_token_count(args, plan):
    """Sample documents for train and eval sets, save as raw text.

    For each language, documents are sampled in deterministic order.
    The first documents go to the train set (up to target_tokens),
    then the next documents go to the eval set (up to eval_tokens).
    This guarantees zero overlap between train and eval.

    To avoid OOM on machines with limited RAM, train texts are flushed
    to disk as shards every --flush-every files. train.py and
    smoke_train.py detect and load these shards automatically.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer_slug = args.tokenizer_name.replace("/", "_")
    data_base = os.path.join(args.data_dir, tokenizer_slug)
    flush_every = args.flush_every

    for lang in plan:
        config = LANG_CONFIG[lang]
        target_tokens = config["target_tokens"]
        eval_tokens = config["eval_tokens"]
        lang_dir = os.path.join(args.raw_dir, lang)
        train_path = os.path.join(data_base, "train", lang)
        eval_path = os.path.join(data_base, "eval", lang)

        train_exists = os.path.exists(train_path)
        eval_exists = os.path.exists(eval_path)
        if train_exists and eval_exists:
            print(f"[{lang}] Already sampled at {train_path} and {eval_path}, skipping.")
            continue

        if not os.path.isdir(lang_dir):
            print(f"[{lang}] Directory {lang_dir} not found. Run 'download' step first.")
            continue

        parquet_files = sorted([
            os.path.join(lang_dir, f)
            for f in os.listdir(lang_dir) if f.endswith(".parquet")
        ])

        if not parquet_files:
            print(f"[{lang}] No parquet files found in {lang_dir}, skipping.")
            continue

        rng = random.Random(f"{SEED}_{lang}")
        shuffled_files = parquet_files.copy()
        rng.shuffle(shuffled_files)

        total_needed = target_tokens + eval_tokens
        print(f"\n[{lang}] Sampling from {len(parquet_files)} files "
              f"(train: {target_tokens / 1e9:.1f}B, eval: {eval_tokens / 1e6:.0f}M tokens)...")

        train_texts = []
        eval_texts = []
        total_tokens = 0
        train_doc_count = 0

        train_shard_idx = 0
        files_since_flush = 0

        for file_idx, filepath in enumerate(shuffled_files):
            if total_tokens >= total_needed:
                break

            filename = os.path.basename(filepath)
            table = pq.read_table(filepath, columns=["text"])
            texts = table.column("text").to_pylist()
            del table

            file_rng = random.Random(f"{SEED}_{lang}_{file_idx}")
            file_rng.shuffle(texts)

            token_counts = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]

            for text, count in zip(texts, token_counts):
                if total_tokens >= total_needed:
                    break
                if count == 0:
                    continue
                if total_tokens < target_tokens:
                    train_texts.append(text)
                else:
                    eval_texts.append(text)
                total_tokens += count

            del texts, token_counts
            phase = "train" if total_tokens < target_tokens else "eval"
            print(f"  [{file_idx + 1}/{len(shuffled_files)}] {filename} "
                  f"→ {total_tokens:,} tokens so far ({phase})")

            files_since_flush += 1
            if not train_exists and train_texts and files_since_flush >= flush_every:
                train_doc_count += len(train_texts)
                _flush_shard(train_texts, train_path, train_shard_idx, "train")
                train_shard_idx += 1
                train_texts = []
                files_since_flush = 0

        # Flush remaining train texts
        if not train_exists and train_texts:
            train_doc_count += len(train_texts)
            _flush_shard(train_texts, train_path, train_shard_idx, "train")
            train_shard_idx += 1
        else:
            train_doc_count += len(train_texts)
        del train_texts

        print(f"  Train: {train_doc_count:,} documents ({train_shard_idx} shards)")
        print(f"  Eval:  {len(eval_texts):,} documents")

        if not eval_texts:
            print(f"  WARNING: No eval data collected — not enough documents after train set")

        if eval_texts and not eval_exists:
            print(f"  Saving eval set to {eval_path}...")
            dataset = Dataset.from_dict({"text": eval_texts})
            dataset.save_to_disk(eval_path)
            del dataset
        del eval_texts

        print(f"  Done: {lang}")

    print(f"\nAll sampling complete.")
    print(f"  Train: {os.path.join(data_base, 'train')}")
    print(f"  Eval:  {os.path.join(data_base, 'eval')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download and prepare CulturaX data")
    parser.add_argument(
        "steps",
        nargs="*",
        default=["download", "sample"],
        help="Steps to run: download, sample, or both (default: download sample)",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory for downloaded parquet files",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base data directory (output: data/{tokenizer}/train and data/{tokenizer}/eval)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show the plan, don't execute",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="gpt2",
        help="Tokenizer for counting tokens (default: gpt2)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=list(LANG_CONFIG.keys()),
        help="Languages to process (default: all)",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Flush train texts to a shard every N parquet files to limit RAM usage (default: 1)",
    )
    args = parser.parse_args()

    # Build plan
    fs = HfFileSystem(token=HF_TOKEN)
    plan = {}

    for lang in args.langs:
        config = LANG_CONFIG[lang]
        all_files = list_parquet_files(fs, lang)
        rng = random.Random(f"{SEED}_{lang}")
        selected = select_files(all_files, config["num_files"], rng)
        plan[lang] = selected

        action = ("all" if config["num_files"] is None or config["num_files"] >= len(all_files)
                  else f"sampled {len(selected)}/{len(all_files)}")
        print(f"[{lang}] {action} files = {len(selected)} "
              f"(target: {config['target_tokens'] / 1e9:.1f}B tokens)")

    print(f"\nTotal files: {sum(len(f) for f in plan.values())}")

    if args.dry_run:
        print("\n[Dry run] No actions taken.")
        for lang, files in plan.items():
            print(f"\n  {lang} ({len(files)} files):")
            for f in files[:5]:
                print(f"    {os.path.basename(f)}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")
        return

    steps = args.steps
    valid_steps = {"download", "sample"}
    invalid = set(steps) - valid_steps
    if invalid:
        parser.error(f"Invalid steps: {invalid}. Choose from: {valid_steps}")

    if "download" in steps:
        run_download(args, plan)

    if "sample" in steps:
        sample_by_token_count(args, plan)


if __name__ == "__main__":
    main()
