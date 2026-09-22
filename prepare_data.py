"""Download, tokenize, and sample CulturaX data for cross-lingual training.

Train targets (in tokens):
  - en: 30B tokens
  - vi, zh, ru, de, ar: 1B tokens each

Eval targets (sampled from documents immediately after train set):
  - en, vi, zh, ru, de, ar: 10M tokens each

Download pipeline:
  1. List all parquet files per language via HfFileSystem
  2. Randomly select files (seeded for reproducibility)
  3. Download selected files via snapshot_download

Offline sampling pipeline:
  1. Validate local parquet paths and sizes against the committed manifest
  2. Load the tokenizer locally
  3. Sample documents, using the tokenizer to count tokens
  4. Continue sampling for eval set (zero overlap with train)
  5. Save raw text as HuggingFace Dataset (Arrow format)

Usage:
  python prepare_data.py --dry-run                    # preview download plan
  python prepare_data.py download                     # download parquet files
  python prepare_data.py sample                       # sample local files by token count
  python prepare_data.py download sample              # both steps
  python prepare_data.py download sample --langs en vi  # specific languages
  python prepare_data.py sample --tokenizer-name meta-llama/Llama-3-8B  # different tokenizer
  python prepare_data.py sample --local-files-only \
      --tokenizer-path /path/to/Qwen3-0.6B             # fully offline
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

# num_files: how many parquet files to download per language.
# At ~2.46GB / ~857M tokens per en file, 50 files ≈ 123GB / ~43B tokens —
# comfortable margin over the 30B target. 5 files per other language provide
# enough data for the 1B+10M target. Selection is deterministic via SEED.
LANG_CONFIG = {
    "en": {"target_tokens": 30_000_000_000, "eval_tokens": 10_000_000, "num_files": 50},
    "vi": {"target_tokens": 1_000_000_000, "eval_tokens": 10_000_000, "num_files": 5},
    "zh": {"target_tokens": 1_000_000_000, "eval_tokens": 10_000_000, "num_files": 5},
    "ru": {"target_tokens": 1_000_000_000, "eval_tokens": 10_000_000, "num_files": 5},
    "de": {"target_tokens": 1_000_000_000, "eval_tokens": 10_000_000, "num_files": 5},
    "ar": {"target_tokens": 1_000_000_000, "eval_tokens": 10_000_000, "num_files": 5},
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


def build_remote_plan(args):
    """Build the deterministic CulturaX download plan from the Hub."""
    fs = HfFileSystem(token=HF_TOKEN or None)
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

    return plan


def load_manifest_entries(manifest_path, langs):
    """Load validated ``language/file.parquet`` entries from the raw manifest."""
    entries = {lang: {} for lang in langs}

    with open(manifest_path, encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue

            fields = line.split("\t")
            if len(fields) != 3:
                raise ValueError(
                    f"Malformed manifest line {line_number}: expected 3 tab-separated fields"
                )

            expected_hash, expected_bytes, relative_path = fields
            path_parts = relative_path.split("/")
            if (len(path_parts) != 2 or path_parts[0] not in LANG_CONFIG
                    or not path_parts[1].endswith(".parquet")):
                raise ValueError(
                    f"Malformed manifest path on line {line_number}: {relative_path}"
                )

            lang = path_parts[0]
            if lang not in entries:
                continue
            if relative_path in entries[lang]:
                raise ValueError(f"Duplicate manifest path: {relative_path}")
            if len(expected_hash) != 64:
                raise ValueError(f"Malformed SHA-256 for {relative_path}")
            try:
                int(expected_hash, 16)
            except ValueError as exc:
                raise ValueError(f"Malformed SHA-256 for {relative_path}") from exc

            try:
                expected_size = int(expected_bytes)
            except ValueError as exc:
                raise ValueError(f"Malformed byte size for {relative_path}") from exc
            if expected_size < 0:
                raise ValueError(f"Negative byte size for {relative_path}")

            entries[lang][relative_path] = {
                "sha256": expected_hash,
                "bytes": expected_size,
            }

    return entries


def build_local_plan(args):
    """Build an offline plan and require the exact manifested local file set."""
    if not os.path.isfile(args.manifest):
        raise FileNotFoundError(f"Raw-data manifest not found: {args.manifest}")

    manifest_entries = load_manifest_entries(args.manifest, args.langs)
    plan = {}

    for lang in args.langs:
        lang_dir = os.path.join(args.raw_dir, lang)
        if not os.path.isdir(lang_dir):
            raise FileNotFoundError(f"Raw language directory not found: {lang_dir}")

        expected = manifest_entries[lang]
        expected_paths = set(expected)
        actual_paths = {
            f"{lang}/{filename}"
            for filename in os.listdir(lang_dir)
            if (filename.endswith(".parquet")
                and os.path.isfile(os.path.join(lang_dir, filename)))
        }

        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise RuntimeError(f"[{lang}] Local parquet set does not match manifest: "
                               + ", ".join(details))

        expected_count = LANG_CONFIG[lang]["num_files"]
        if expected_count is not None and len(expected_paths) != expected_count:
            raise RuntimeError(
                f"[{lang}] Manifest has {len(expected_paths)} files; expected {expected_count}"
            )

        local_files = []
        for relative_path in sorted(expected_paths):
            local_path = os.path.join(args.raw_dir, relative_path)
            actual_bytes = os.path.getsize(local_path)
            expected_bytes = expected[relative_path]["bytes"]
            if actual_bytes != expected_bytes:
                raise RuntimeError(
                    f"[{lang}] Size mismatch for {relative_path}: "
                    f"expected {expected_bytes}, got {actual_bytes}"
                )
            local_files.append(local_path)

        plan[lang] = local_files
        print(f"[{lang}] verified local files = {len(local_files)} "
              f"(target: {LANG_CONFIG[lang]['target_tokens'] / 1e9:.1f}B tokens)")

    return plan


def run_download(args, plan):
    for lang, files in plan.items():
        filenames = [os.path.basename(f) for f in files]
        allow_patterns = [f"{lang}/{fn}" for fn in filenames]

        print(f"\n[{lang}] Downloading {len(files)} files...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN or None,
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
    to disk as shards every --flush-every files. train.py detects
    and loads these shards automatically.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer_source = args.tokenizer_path or args.tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=args.local_files_only,
    )
    tokenizer_slug = args.tokenizer_name.replace("/", "_")
    data_base = os.path.join(args.data_dir, tokenizer_slug)
    flush_every = args.flush_every
    tokenize_batch_size = args.tokenize_batch_size

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
        if train_exists or eval_exists:
            raise RuntimeError(
                f"[{lang}] Partial output detected: train_exists={train_exists}, "
                f"eval_exists={eval_exists}. Remove both language outputs and rerun."
            )

        parquet_files = sorted(
            os.path.join(lang_dir, os.path.basename(filepath))
            for filepath in plan[lang]
        )
        missing_files = [filepath for filepath in parquet_files if not os.path.isfile(filepath)]
        if missing_files:
            raise FileNotFoundError(f"[{lang}] Missing planned files: {missing_files}")

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

            for batch_start in range(0, len(texts), tokenize_batch_size):
                if total_tokens >= total_needed:
                    break
                batch_texts = texts[batch_start:batch_start + tokenize_batch_size]
                batch_input_ids = tokenizer(
                    batch_texts,
                    add_special_tokens=False,
                )["input_ids"]
                if len(batch_input_ids) != len(batch_texts):
                    raise RuntimeError(
                        f"[{lang}] Tokenizer returned {len(batch_input_ids)} results "
                        f"for a batch of {len(batch_texts)} documents"
                    )

                for text, input_ids in zip(batch_texts, batch_input_ids):
                    if total_tokens >= total_needed:
                        break
                    count = len(input_ids)
                    if count == 0:
                        continue
                    if total_tokens < target_tokens:
                        train_texts.append(text)
                    else:
                        eval_texts.append(text)
                    total_tokens += count

                del batch_texts, batch_input_ids

            del texts
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

        if total_tokens < total_needed:
            raise RuntimeError(
                f"[{lang}] Insufficient data: sampled {total_tokens:,} tokens, "
                f"need {total_needed:,}"
            )

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
        default="Qwen/Qwen3-0.6B",
        help=("Logical tokenizer name used for the output directory "
              "(default: Qwen/Qwen3-0.6B)"),
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help=("Optional local tokenizer/model directory to load while retaining "
              "--tokenizer-name for the output directory"),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Forbid tokenizer downloads and use only local tokenizer files",
    )
    parser.add_argument(
        "--manifest",
        default="resources/culturax_raw_manifest.tsv",
        help="Manifest used to validate local parquet paths and sizes in sample-only mode",
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
    parser.add_argument(
        "--tokenize-batch-size",
        type=int,
        default=4096,
        help="Documents per tokenizer call during sampling (default: 4096)",
    )
    args = parser.parse_args()

    steps = args.steps
    valid_steps = {"download", "sample"}
    invalid = set(steps) - valid_steps
    if invalid:
        parser.error(f"Invalid steps: {invalid}. Choose from: {valid_steps}")

    invalid_langs = set(args.langs) - set(LANG_CONFIG)
    if invalid_langs:
        parser.error(f"Unknown languages: {sorted(invalid_langs)}")
    if args.flush_every <= 0:
        parser.error("--flush-every must be positive")
    if args.tokenize_batch_size <= 0:
        parser.error("--tokenize-batch-size must be positive")
    if args.num_workers <= 0:
        parser.error("--num-workers must be positive")

    # Download requires a Hub listing. Sample-only mode is strictly local.
    if "download" in steps:
        plan = build_remote_plan(args)
    else:
        plan = build_local_plan(args)

    print(f"\nTotal files: {sum(len(files) for files in plan.values())}")

    if args.dry_run:
        print("\n[Dry run] No actions taken.")
        for lang, files in plan.items():
            print(f"\n  {lang} ({len(files)} files):")
            for filepath in files[:5]:
                print(f"    {os.path.basename(filepath)}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")
        return

    if "download" in steps:
        run_download(args, plan)

    if "sample" in steps:
        sample_by_token_count(args, plan)


if __name__ == "__main__":
    main()
