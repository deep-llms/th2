#!/usr/bin/env python3
"""End-to-end smoke of the product-code arm through the real entry points.

Builds tiny fixtures (a 1-layer Qwen3 config with the *real* 151,936 vocab, the
Qwen3 tokenizer, two small per-language datasets, a fake dense table), then
runs, via subprocess:

  1. train_compositional.py  --arm product_code (hashed), fresh, N steps
  2. the same command with more steps      -> automatic checkpoint resume
  3. eval/ppl_bytoken.py                     -> verification gap must be small
  4. eval/eval_checkpoint.py --ppl-only
  5. scripts/make_pq_codes.py on the fake dense table
  6. train_compositional.py  --arm product_code --product_code_assignment pq

With --gpus N > 1, steps 1-2 run under `accelerate launch --multi_gpu` (DDP
with find_unused_parameters=False, per-rank RNG files, DDP resume). Logs and a
summary JSON are written under --scratch so the evidence persists.

Never run run_experiments.py for this (its pre-flight kills GPU processes).

Usage:
  python scripts/smoke_product_code_e2e.py --scratch temp/product_code_smoke --gpus 4
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


def make_fixtures(scratch, tokenizer_name, hidden):
    import torch
    from datasets import Dataset
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Qwen3Config

    cfg_dir = scratch / "cfg"
    Qwen3Config(
        vocab_size=151936, hidden_size=hidden, intermediate_size=2 * hidden,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=hidden // 2, max_position_embeddings=256,
        tie_word_embeddings=False,
    ).save_pretrained(cfg_dir)
    AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(cfg_dir)
    rng = random.Random(0)
    words = ["the", "river", "bank", "account", "ngân", "hàng", "银行", "Fluss",
             "деньги", "مصرف", "quantum", "token", "code", "zebra", "über", "东京"]

    def docs(n):
        return [" ".join(rng.choice(words) for _ in range(rng.randint(20, 60)))
                for _ in range(n)]

    for split, n in (("train", 300), ("eval", 40)):
        for lang in ("en", "zh"):
            Dataset.from_dict({"text": docs(n)}).save_to_disk(str(scratch / split / lang))
    (scratch / "fake_dense").mkdir(exist_ok=True)
    save_file({"model.embed_tokens.weight": torch.randn(151936, hidden)},
              str(scratch / "fake_dense" / "model.safetensors"))


def run(name, cmd, scratch, results, env=None):
    log = scratch / f"{name}.log"
    start = time.time()
    with open(log, "w") as handle:
        code = subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT,
                               cwd=str(PROJECT_ROOT), env=env)
    results[name] = {"exit": code, "seconds": round(time.time() - start, 1),
                     "log": str(log)}
    print(f"[{'PASS' if code == 0 else 'FAIL'}] {name} ({results[name]['seconds']}s) -> {log}")
    return code == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scratch", default="temp/product_code_smoke")
    parser.add_argument("--gpus", type=int, default=1, help=">1 uses accelerate --multi_gpu")
    parser.add_argument("--tokenizer", default=os.environ.get("SPARSE_EMB_MODEL_DIR", "Qwen/Qwen3-0.6B"))
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--skip-pq", action="store_true")
    parser.add_argument("--port", type=int, default=29517)
    args = parser.parse_args()

    scratch = Path(args.scratch).resolve()
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    make_fixtures(scratch, args.tokenizer, args.hidden)
    py = sys.executable
    env = dict(os.environ, WANDB_MODE="offline", NCCL_NVLS_ENABLE="0")
    env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(args.gpus)))

    launcher = [py]
    if args.gpus > 1:
        launcher = [py, "-m", "accelerate.commands.launch", "--num_processes", str(args.gpus),
                    "--multi_gpu", "--main_process_port", str(args.port)]
    common = [
        "train_compositional.py", "--config_name", str(scratch / "cfg"),
        "--tokenizer_name", str(scratch / "cfg"), "--data_dir", str(scratch / "train"),
        "--block_size", "64", "--seed", "42", "--bf16", "--ddp_timeout", "600",
        "--ddp_find_unused_parameters", "false", "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "2", "--learning_rate", "3e-4",
        "--lr_scheduler_type", "cosine_with_min_lr", "--lr_scheduler_kwargs", '{"min_lr_rate":0.1}',
        "--warmup_steps", "1", "--weight_decay", "0.1", "--adam_beta1", "0.9", "--adam_beta2", "0.95",
        "--max_grad_norm", "1.0", "--logging_steps", "1", "--save_steps", "3",
        "--dataloader_num_workers", "0", "--report_to", "none",
        "--arm", "product_code", "--product_code_head_size", "2048",
        "--product_code_num_hashes", "4", "--product_code_num_buckets", "4096",
        "--product_code_importance_path", "resources/token_importance_langbalanced.npz",
        "--product_code_importance_key", "counts", "--product_code_seed", "0", "--tie_output",
    ]
    results = {}
    ok = True
    out = scratch / "out_hashed"
    ok &= run("train_fresh", launcher + common + ["--output_dir", str(out), "--max_steps", "6",
              "--product_code_assignment", "hashed"], scratch, results, env)
    ok &= run("train_resume", launcher + common + ["--output_dir", str(out), "--max_steps", "9",
              "--product_code_assignment", "hashed"], scratch, results, env)
    ck = out / "checkpoint-9"
    if ok:
        import torch
        emb = torch.load(ck / "embedding.pt", map_location="cpu", weights_only=True)
        moved = float((emb["gate_offsets"] != 0).float().mean())
        unique = int(torch.unique(emb["codes"], dim=0).size(0)) == int(emb["codes"].size(0))
        rng_files = sorted(p.name for p in ck.glob("rng_state*"))
        results["checkpoint"] = {"gate_offsets_nonzero_fraction": moved, "codes_unique": unique,
                                 "rng_files": rng_files}
        print(f"[{'PASS' if moved > 0.9 and unique else 'FAIL'}] checkpoint: gate offsets moved={moved:.3f}, "
              f"codes unique={unique}, rng files={rng_files}")
        ok &= moved > 0.9 and unique
    single = dict(env, CUDA_VISIBLE_DEVICES=env["CUDA_VISIBLE_DEVICES"].split(",")[0])
    ok &= run("ppl_bytoken", [py, "eval/ppl_bytoken.py", "--checkpoint", str(ck), "--eval-dir",
              str(scratch / "eval"), "--tokenizer-name", str(scratch / "cfg"), "--bf16",
              "--block-size", "64", "--output-dir", str(scratch / "bytoken")], scratch, results, single)
    ok &= run("eval_checkpoint", [py, "eval/eval_checkpoint.py", "--checkpoint", str(ck), "--eval-dir",
              str(scratch / "eval"), "--tokenizer-name", str(scratch / "cfg"), "--bf16", "--ppl-only",
              "--block-size", "64", "--output-dir", str(scratch / "eval_out")], scratch, results, single)
    if not args.skip_pq:
        ok &= run("make_pq_codes", [py, "scripts/make_pq_codes.py", "--checkpoint", str(scratch / "fake_dense"),
                  "--head-size", "2048", "--num-hashes", "4", "--num-buckets", "4096", "--iters", "2",
                  "--device", "cuda", "--output", str(scratch / "pq_codes.pt")], scratch, results, single)
        ok &= run("train_pq", launcher + common + ["--output_dir", str(scratch / "out_pq"), "--max_steps", "3",
                  "--product_code_assignment", "pq", "--product_code_codes_path", str(scratch / "pq_codes.pt")],
                  scratch, results, env)
    results["overall"] = "PASS" if ok else "FAIL"
    results["gpus"] = args.gpus
    with open(scratch / "summary.json", "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"overall: {results['overall']} (summary: {scratch / 'summary.json'})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
