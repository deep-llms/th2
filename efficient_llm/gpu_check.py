"""Real pretrained-model identity, causal-prefix and cached-decode verification."""
import argparse
from pathlib import Path
import torch

from efficient_llm.diagnostic import load_model, load_probe, write
from efficient_llm.group_selection import GroupSelection, subset_masks, contributions, reconstruction_errors, apply_mask


@torch.inference_mode()
def check(model_path, probe, output):
    if output.exists():
        raise ValueError("GPU verification output already exists")
    manifest, tokens = load_probe(probe)
    torch.set_float32_matmul_precision("highest")
    # FP32 checks isolate implementation differences from BF16 discontinuities
    # in argmax subset selection. Production BF16 identity/causality is checked below.
    model = load_model(model_path, "cuda:0", torch.float32)
    ids = tokens["evaluation"][:1, :24].cuda()
    groups = model.config.num_key_value_heads
    keep = groups-2
    static = subset_masks(groups, keep, device=ids.device)[0].expand(model.config.num_hidden_layers, -1)
    results = {}
    full_reference = model(ids, use_cache=False).logits
    for policy in ("dense", "static", "raw_norm", "projected_norm", "joint"):
        with GroupSelection(model, keep=keep, policy=policy, static_masks=static):
            full = model(ids, use_cache=False).logits
            altered = ids.clone(); altered[:, 12:] = 1
            causal = model(altered, use_cache=False).logits
            torch.testing.assert_close(causal[:, :12], full[:, :12], rtol=2e-5, atol=2e-5)
            cache, parts = None, []
            for t in range(ids.shape[1]):
                out = model(ids[:, t:t+1], past_key_values=cache, use_cache=True)
                parts.append(out.logits); cache = out.past_key_values
            cached = torch.cat(parts, 1)
            torch.testing.assert_close(cached, full, rtol=3e-4, atol=2e-3)
            if cache.get_seq_length() != ids.shape[1] or any(layer.keys.shape[1] != groups for layer in cache.layers):
                raise ValueError("Dense KV history was not preserved")
            results[policy] = {"fp32_cached_max_abs": (cached-full).abs().max().item(),
                               "fp32_causal_max_abs": (causal[:, :12]-full[:, :12]).abs().max().item()}
    for policy in ("dense", "raw_norm", "projected_norm", "joint"):
        with GroupSelection(model, keep=groups, policy=policy):
            torch.testing.assert_close(model(ids, use_cache=False).logits, full_reference, rtol=0, atol=0)
    # Independently verify the objective against a direct masked O projection.
    weight = model.model.layers[0].self_attn.o_proj.weight
    x = torch.randn(5, weight.shape[1], device=weight.device)
    c = contributions(x, weight, groups)
    masks = subset_masks(groups, keep, device=weight.device)
    errors = reconstruction_errors(c, masks)
    for i, mask in enumerate(masks):
        direct = torch.nn.functional.linear(x-apply_mask(x, mask), weight).square().sum(-1)
        torch.testing.assert_close(errors[:, i], direct, rtol=3e-5, atol=3e-4)
    model.bfloat16()
    full_reference = model(ids, use_cache=False).logits
    for policy in ("raw_norm", "projected_norm", "joint"):
        with GroupSelection(model, keep=groups, policy=policy):
            torch.testing.assert_close(model(ids, use_cache=False).logits, full_reference, rtol=0, atol=0)
    for policy in results:
        with GroupSelection(model, keep=keep, policy=policy, static_masks=static):
            full = model(ids, use_cache=False).logits
            altered = ids.clone(); altered[:, 12:] = 1
            prefix = model(altered, use_cache=False).logits[:, :12]
            torch.testing.assert_close(prefix, full[:, :12], rtol=0, atol=0)
            if not torch.isfinite(full).all():
                raise ValueError("Nonfinite BF16 logits")
            results[policy]["bf16_causal_max_abs"] = (prefix-full[:, :12]).abs().max().item()
    write(output, {"status": "passed", "model": str(model_path), "gpu": torch.cuda.get_device_name(0),
        "checks": results, "keep_all_exact_identity": True, "all_kv_groups_preserved": True,
        "probe_tokens_sha256": manifest["tokens_sha256"],
        "scope": "FP32 cached equivalence; BF16 keep-all identity, causality and finite logits. No sparse kernel tested."})
    print("GPU_CHECK_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.model, args.probe, args.output)
