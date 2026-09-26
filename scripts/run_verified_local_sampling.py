"""Run the captured B200 sampler offline after local download/env verification.

The run directory holds a frozen source/, tokenizer/, data/, B200 environment
reference, download receipt, and verify_environment.py. Run with the matching
sampling environment. Never resumes or overwrites partially sampled outputs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--download-session", required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="true")
    output = root / "data/Qwen_Qwen3-0.6B-Base"
    if output.exists() or (root / "sampling_started.json").exists():
        raise ValueError("Sampled output/run already exists; refusing partial resume")
    try:
        while not (root / "download_complete.json").exists():
            state = subprocess.run(
                ["tmux", "display-message", "-p", "-t", args.download_session, "#{pane_dead}"],
                capture_output=True, text=True)
            if state.returncode or state.stdout.strip() != "0":
                raise RuntimeError("Download exited without a verified receipt; inspect download.log")
            print("Waiting for download and all 75 SHA256 checks", flush=True)
            time.sleep(20)
        receipt = json.loads((root / "download_complete.json").read_text())
        if receipt.get("status") != "ok" or receipt.get("files") != 75:
            raise ValueError("Invalid download receipt")
        subprocess.run([sys.executable, "-u", str(root / "verify_environment.py")], check=True)
        argv = [sys.executable, "-u", str(root / "source/prepare_data.py"), "sample",
                "--raw-dir", str(root / "data/raw"), "--data-dir", str(root / "data"),
                "--manifest", str(root / "source/culturax_raw_manifest.tsv"),
                "--tokenizer-name", "Qwen/Qwen3-0.6B-Base",
                "--tokenizer-path", str(root / "tokenizer"), "--local-files-only",
                "--langs", "en", "vi", "zh", "ru", "de", "ar",
                "--flush-every", "1", "--tokenize-batch-size", "4096"]
        subprocess.run([*argv, "--dry-run"], check=True)
        write_json(root / "sampling_started.json", {
            "started_utc": datetime.now(timezone.utc).isoformat(), "argv": argv,
            "python": sys.executable, "output": str(output)})
        with (root / "sampling.log").open("x") as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)

        from datasets import load_from_disk
        languages = {}
        for lang in ("en", "vi", "zh", "ru", "de", "ar"):
            splits = {}
            for split in ("train", "eval"):
                directory = output / split / lang
                shards = sorted(directory.glob("shard_*")) if split == "train" else [directory]
                if not shards:
                    raise ValueError(f"Missing shards: {directory}")
                count = 0
                digest = hashlib.sha256()
                for shard in shards:
                    ds = load_from_disk(str(shard))
                    if ds.column_names != ["text"] or not len(ds):
                        raise ValueError(f"Invalid sampled shard: {shard}")
                    for batch in ds.iter(batch_size=4096):
                        for text in batch["text"]:
                            encoded = text.encode("utf-8")
                            digest.update(struct.pack("<Q", len(encoded)))
                            digest.update(encoded)
                            count += 1
                splits[split] = {"documents": count, "shards": len(shards),
                                 "ordered_text_sha256": digest.hexdigest()}
            languages[lang] = splits
            print("VERIFIED_SAMPLED_LANGUAGE", lang, json.dumps(splits), flush=True)
        write_json(root / "sampling_complete.json", {
            "success": True, "completed_utc": datetime.now(timezone.utc).isoformat(),
            "output": str(output), "languages": languages,
            "output_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
            "ordered_text_hash_format": "UTF-8 documents, each prefixed by uint64 little-endian byte length",
            "b200_full_output_comparison": "pending"})
        print("LOCAL_SAMPLING_COMPLETE", flush=True)
    except BaseException as error:
        write_json(root / "sampling_failure.json", {
            "utc": datetime.now(timezone.utc).isoformat(), "error": str(error)})
        raise


if __name__ == "__main__":
    main()
