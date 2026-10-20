"""Optional real-kernel SDPA/FA4 comparison; run only on an authorized free GPU.

python tests/ml/fa4_gpu_probe.py /absolute/new-receipt.json
No downloads, process management, environment installation or training job launch.
This is a tiny numerical smoke check, not long-training or throughput evidence.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from packed_training import PackedCollator, model_inputs
from packed_fa4 import register_backend, FA4_VERSION


def relative_error(actual, expected):
    return float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("fa4", "fa2"), default="fa4")
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise ValueError("Choose a fresh receipt path")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This probe requires an authorized CUDA GPU supporting BF16")
    from importlib import import_module
    backend = import_module("packed_" + args.backend).register_backend()
    torch.manual_seed(42)
    rows = []
    for boundary in (17, 31):
        ids = torch.randint(0, 128, (96,)).tolist()
        rows.append(dict(input_ids=ids, labels=ids, attention_mask=[1] * 96,
                         segments=[0] * boundary + [1] * (96 - boundary)))

    def inputs(backend, rows):
        batch = PackedCollator(attention_backend=backend)(rows)
        return model_inputs({k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()})

    records = []
    for checkpointing in (False, True):
        config = Qwen3Config(vocab_size=128, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=64, max_position_embeddings=128, attention_dropout=0., use_cache=False)
        dense = Qwen3ForCausalLM._from_config(config, attn_implementation="sdpa").cuda().train()
        fa4 = copy.deepcopy(dense)
        fa4.set_attn_implementation(backend)
        if checkpointing:
            for model in (dense, fa4):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, b = dense(**inputs("sdpa", rows)), fa4(**inputs(args.backend, rows))
        a.loss.backward()
        b.loss.backward()
        grads_a = torch.cat([p.grad.flatten() for p in dense.parameters()])
        grads_b = torch.cat([p.grad.flatten() for p in fa4.parameters()])
        record = dict(checkpointing=checkpointing, sdpa_loss=float(a.loss.detach()),
            fa4_loss=float(b.loss.detach()), logits_relative_l2=relative_error(b.logits.detach(), a.logits.detach()),
            gradients_relative_l2=relative_error(grads_b, grads_a))
        if not torch.isfinite(grads_b).all() or not torch.isfinite(b.logits).all():
            raise AssertionError(f"Nonfinite FA4 results: {record}")
        # Explicit BF16 smoke tolerances, not a long-run equivalence claim.
        if record["logits_relative_l2"] > .02 or record["gradients_relative_l2"] > .05:
            raise AssertionError(f"Backend mismatch: {record}")
        changed = copy.deepcopy(rows)
        changed[0]["input_ids"] = [(x + 7) % 128 for x in rows[0]["input_ids"][:17]] + rows[0]["input_ids"][17:]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            after = fa4(**inputs(args.backend, changed)).logits
        torch.testing.assert_close(b.logits[0, 17:], after[0, 17:], rtol=0, atol=0)
        torch.testing.assert_close(b.logits[1], after[1], rtol=0, atol=0)
        records.append(record)
        del dense, fa4, a, b, grads_a, grads_b
    result = dict(status="passed", torch=torch.__version__, backend=args.backend,
        kernel_version=__import__("importlib.metadata").metadata.version("flash-attn-4" if args.backend == "fa4" else "flash-attn"),
        gpu=torch.cuda.get_device_name(), dtype="bf16 autocast, FP32 weights",
        note="Tiny real-kernel smoke only; not throughput or long-run validation", checks=records)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
