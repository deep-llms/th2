"""Eight-process CPU regression: interrupted vs uninterrupted four-arm training.

Run with torchrun --standalone --nproc-per-node=8 and a fresh output directory.
Uses synthetic Hugging Face datasets and the native Trainer sampler. No GPU use.
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
from deep_kv.__main__ import build_identity, offline, setup_distributed, smoke_inputs
from deep_kv.config import Recipe
from deep_kv.model import DeepKV
from deep_kv.training import train, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    offline()
    torch.set_num_threads(1)
    setup_distributed(cpu=True)
    try:
        if dist.get_rank() == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        recipe, data, dev = smoke_inputs()
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                            max_position_embeddings=64, attention_dropout=0.0, tie_word_embeddings=True)
        config._attn_implementation = "sdpa"
        results = {}
        for arm in "ABCD":
            observed_dtypes = set()

            def check_projection_dtype(module, inputs, output):
                observed_dtypes.add(str(output.dtype))
                expected = torch.bfloat16 if args.bf16 else torch.float32
                assert output.dtype == expected, (arm, output.dtype, expected)

            def fresh():
                result = DeepKV.from_scratch(copy.deepcopy(config), arm, consumer=recipe.consumer,
                                            deep_target=recipe.deep_target, lm_chunk=recipe.lm_chunk)
                result.backbone.model.layers[0].self_attn.q_proj.register_forward_hook(check_projection_dtype)
                return result
            full = fresh()
            identity = build_identity({"synthetic_resume_check": True}, recipe, full, {"fingerprint": data._fingerprint}, mixed_precision=args.bf16)
            train(full, data, dev, recipe, args.output / arm / "full", identity, microbatch=16, mixed_precision=args.bf16)
            train(fresh(), data, dev, recipe, args.output / arm / "resumed", identity,
                  microbatch=16, mixed_precision=args.bf16, stop_after=2)
            resumed = fresh()
            train(resumed, data, dev, recipe, args.output / arm / "resumed", identity,
                  microbatch=16, mixed_precision=args.bf16, resume=True)
            trainer_state = json.loads((args.output / arm / "resumed" /
                                        f"checkpoint-{recipe.updates}/trainer_state.json").read_text())
            for row in trainer_state["log_history"]:
                if "eval_objective" in row:
                    assert row["eval_loss"] == row["eval_objective"]
            for p, q in zip(full.parameters(), resumed.parameters()):
                torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)
            delta = max((p - q).abs().max().item() for p, q in zip(full.parameters(), resumed.parameters()))
            maximum = torch.tensor(delta, dtype=torch.float64)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            assert observed_dtypes
            results[arm] = {"maximum_parameter_difference": maximum.item(), "updates": recipe.updates,
                            "projection_dtypes": sorted(observed_dtypes)}
        if dist.get_rank() == 0:
            write_json(args.output / "resume_verified.json", {"status": "ok", "cpu_only": True,
                                                              "bf16": args.bf16, "microbatch": 16,
                                                              "world_size": dist.get_world_size(), "arms": results})
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
