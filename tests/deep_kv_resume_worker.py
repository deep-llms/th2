"""Eight-process CPU regression: interrupted vs uninterrupted four-arm training.

Run with torchrun --standalone --nproc-per-node=8, passing an existing synthetic
smoke's prepared directory and a fresh output directory. No GPU use.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from transformers import Qwen3Config
from deep_kv.__main__ import build_identity, offline, setup_distributed
from deep_kv.config import Recipe
from deep_kv.data import TokenStream
from deep_kv.model import DeepKV
from deep_kv.training import train, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    offline()
    torch.set_num_threads(1)
    setup_distributed(cpu=True)
    try:
        if dist.get_rank() == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        recipe = Recipe(**json.loads((args.data_dir / "complete.json").read_text())["recipe"])
        data, dev = (TokenStream(args.data_dir, split, recipe) for split in ("train", "eval"))
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                            max_position_embeddings=64, attention_dropout=0.0, tie_word_embeddings=True)
        config._attn_implementation = "sdpa"
        results = {}
        for arm in "ABCD":
            def fresh():
                return DeepKV.from_scratch(copy.deepcopy(config), arm, consumer=recipe.consumer,
                                           deep_target=recipe.deep_target, lm_chunk=recipe.lm_chunk)
            full = fresh()
            identity = build_identity({"synthetic_resume_check": True}, recipe, full, data.manifest, mixed_precision=False)
            train(full, data, dev, recipe, args.output / arm / "full", identity, mixed_precision=False)
            train(fresh(), data, dev, recipe, args.output / arm / "resumed", identity,
                  mixed_precision=False, stop_after=1)
            resumed = fresh()
            train(resumed, data, dev, recipe, args.output / arm / "resumed", identity,
                  mixed_precision=False, resume=True)
            for p, q in zip(full.parameters(), resumed.parameters()):
                torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)
            delta = max((p - q).abs().max().item() for p, q in zip(full.parameters(), resumed.parameters()))
            maximum = torch.tensor(delta, dtype=torch.float64)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            results[arm] = {"maximum_parameter_difference": maximum.item(), "updates": recipe.updates}
        if dist.get_rank() == 0:
            write_json(args.output / "resume_verified.json", {"status": "ok", "cpu_only": True,
                                                              "world_size": dist.get_world_size(), "arms": results})
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
