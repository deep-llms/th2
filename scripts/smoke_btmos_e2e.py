#!/usr/bin/env python3
"""End-to-end BT-MoS smoke through train, strict resume, and both PPL loaders.

The fixture keeps Qwen3's real 151,936-token vocabulary but uses a one-layer,
128-wide backbone and reduced GroupReduce ranks. With ``--gpus N`` greater
than one, fresh and resumed training run through Accelerate DDP with unused
parameter detection disabled. This script never invokes ``run_experiments``
and never kills GPU processes.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 151_936


def make_fixtures(scratch, tokenizer_name, hidden):
    import numpy as np
    from datasets import Dataset
    from transformers import AutoTokenizer, Qwen3Config

    config_dir = scratch / "cfg"
    Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=hidden // 2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    ).save_pretrained(config_dir)
    AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(config_dir)

    generator = random.Random(0)
    words = [
        "the", "river", "bank", "account", "ngân", "hàng", "银行",
        "Fluss", "деньги", "مصرف", "quantum", "token", "zebra", "東京",
    ]
    for split, size in (("train", 240), ("eval", 32)):
        for language in ("en", "zh"):
            documents = [
                " ".join(generator.choice(words) for _ in range(48))
                for _ in range(size)
            ]
            Dataset.from_dict({"text": documents}).save_to_disk(
                str(scratch / split / language)
            )
    # Stable (-importance, token id) produces deterministic contiguous tiers.
    np.savez(scratch / "importance.npz", counts=np.ones(VOCAB_SIZE))


def run(name, command, scratch, results, environment):
    log_path = scratch / f"{name}.log"
    start = time.time()
    with open(log_path, "w") as handle:
        code = subprocess.call(
            command,
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    results[name] = {
        "exit": code,
        "seconds": round(time.time() - start, 1),
        "log": str(log_path),
    }
    print(
        f"[{'PASS' if code == 0 else 'FAIL'}] {name} "
        f"({results[name]['seconds']}s) -> {log_path}"
    )
    return code == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scratch", default="temp/btmos_smoke")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--tokenizer",
        default=os.environ.get("SPARSE_EMB_MODEL_DIR", "Qwen/Qwen3-0.6B"),
    )
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--port", type=int, default=29523)
    args = parser.parse_args()
    if args.gpus <= 0:
        raise SystemExit("--gpus must be positive")

    scratch = Path(args.scratch).resolve()
    if scratch == PROJECT_ROOT or PROJECT_ROOT not in scratch.parents:
        raise SystemExit("--scratch must be a subdirectory of the project")
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    make_fixtures(scratch, args.tokenizer, args.hidden)

    python = sys.executable
    environment = dict(
        os.environ,
        WANDB_MODE="offline",
        NCCL_NVLS_ENABLE="0",
    )
    environment.setdefault(
        "CUDA_VISIBLE_DEVICES", ",".join(str(index) for index in range(args.gpus))
    )
    launcher = [python]
    if args.gpus > 1:
        launcher = [
            python, "-m", "accelerate.commands.launch",
            "--num_processes", str(args.gpus), "--multi_gpu",
            "--main_process_port", str(args.port),
        ]

    common = [
        "train_compositional.py",
        "--config_name", str(scratch / "cfg"),
        "--tokenizer_name", str(scratch / "cfg"),
        "--data_dir", str(scratch / "train"),
        "--block_size", "64",
        "--preprocessing_num_workers", "1",
        "--seed", "42",
        "--bf16",
        "--ddp_timeout", "600",
        "--ddp_find_unused_parameters", "false",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "2",
        "--learning_rate", "3e-4",
        "--lr_scheduler_type", "cosine_with_min_lr",
        "--lr_scheduler_kwargs", '{"min_lr_rate":0.1}',
        "--warmup_steps", "1",
        "--weight_decay", "0.1",
        "--logging_steps", "1",
        "--save_steps", "3",
        "--dataloader_num_workers", "0",
        "--report_to", "none",
        "--arm", "groupreduce",
        "--groupreduce_num_groups", "4",
        "--groupreduce_ranks", "128,64,32,16",
        "--groupreduce_populations", "2048,6144,24576,119168",
        "--groupreduce_frequency_path", str(scratch / "importance.npz"),
        "--groupreduce_frequency_key", "counts",
        "--mos_components", "3",
        "--mos_context_rank", "32",
        "--mos_chunk_size", "32",
        "--allow_from_scratch_baseline_init",
        "--tie_output",
    ]
    results = {}
    output_dir = scratch / "out"
    ok = run(
        "train_fresh",
        launcher + common + ["--output_dir", str(output_dir), "--max_steps", "3"],
        scratch, results, environment,
    )
    if ok:
        ok = run(
            "train_resume",
            launcher + common + [
                "--output_dir", str(output_dir), "--max_steps", "6"
            ],
            scratch, results, environment,
        )
    checkpoint_dir = output_dir / "checkpoint-6"
    if ok:
        with open(checkpoint_dir / "trainer_state.json") as handle:
            history = json.load(handle)["log_history"]
        metric_keys = {
            key for entry in history for key in entry if key.startswith("mos_prior_")
        }
        expected_metrics = {
            "mos_prior_usage_0", "mos_prior_usage_1", "mos_prior_usage_2",
            "mos_prior_entropy",
        }
        ok = metric_keys == expected_metrics
        results["metrics"] = sorted(metric_keys)
        print(f"[{'PASS' if ok else 'FAIL'}] prior metrics: {sorted(metric_keys)}")

    single_gpu_environment = dict(
        environment,
        CUDA_VISIBLE_DEVICES=environment["CUDA_VISIBLE_DEVICES"].split(",")[0],
    )
    if ok:
        ok = run(
            "ppl_bytoken",
            [
                python, "eval/ppl_bytoken.py", "--checkpoint", str(checkpoint_dir),
                "--eval-dir", str(scratch / "eval"), "--tokenizer-name",
                str(scratch / "cfg"), "--bf16", "--block-size", "64",
                "--output-dir", str(scratch / "bytoken"),
            ],
            scratch, results, single_gpu_environment,
        ) and ok
        ok = run(
            "eval_checkpoint",
            [
                python, "eval/eval_checkpoint.py", "--checkpoint",
                str(checkpoint_dir), "--eval-dir", str(scratch / "eval"),
                "--tokenizer-name", str(scratch / "cfg"), "--bf16",
                "--ppl-only", "--block-size", "64", "--output-dir",
                str(scratch / "eval_out"),
            ],
            scratch, results, single_gpu_environment,
        ) and ok

    results["overall"] = "PASS" if ok else "FAIL"
    results["gpus"] = args.gpus
    with open(scratch / "summary.json", "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"overall: {results['overall']} (summary: {scratch / 'summary.json'})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
