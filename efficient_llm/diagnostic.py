"""Offline, frozen-backbone GQA subset diagnostic; dense execution, no speed claims."""
import os
os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from efficient_llm.diagnostic_data import prepare_probe, sha256
from efficient_llm.group_selection import GroupSelection, subset_masks


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def next_token_sums(logits, targets, chunk=128):
    """No boundary shift here: inputs=tokens[:-1], targets=tokens[1:]."""
    if logits.shape[:2] != targets.shape or targets.ndim != 2:
        raise ValueError("Logits/next-token target shapes differ")
    result = torch.zeros(targets.shape[0], device=logits.device, dtype=torch.float64)
    for start in range(0, targets.shape[1], chunk):
        scores = logits[:, start:start+chunk].float()
        losses = torch.nn.functional.cross_entropy(scores.reshape(-1, scores.shape[-1]),
            targets[:, start:start+chunk].reshape(-1), reduction="none")
        result += losses.reshape(targets.shape[0], -1).double().sum(-1)
    return result


def paired_summary(candidate, reference, seed=42):
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    if difference.ndim != 1 or len(difference) < 2 or not np.isfinite(difference).all():
        raise ValueError("Need finite paired per-document NLLs")
    indices = np.random.default_rng(seed).integers(0, len(difference), (10000, len(difference)))
    means = difference[indices].mean(1)
    return {"delta_nats_per_token": float(difference.mean()),
            "paired_document_bootstrap_95ci": np.quantile(means, [.025, .975]).tolist(),
            "documents_improved": int((difference < 0).sum()), "documents": len(difference)}


def reduce_stats(selection):
    for layer in selection.stats.values():
        for tensor in layer.values():
            if dist.is_initialized():
                dist.all_reduce(tensor)
    return selection.stats


@torch.inference_mode()
def evaluate(model, tokens, *, keep, policy, static_masks=None, collect=False,
             batch_size=2, stride=16, rank=0, world=1, selection=None):
    device = next(model.parameters()).device
    sums = torch.zeros(len(tokens), dtype=torch.float64, device=device)
    coverage = torch.zeros_like(sums)
    indices = list(range(rank, len(tokens), world))
    if not indices:
        raise ValueError("Each rank needs at least one document")
    context = selection if selection is not None else GroupSelection(model, keep=keep, policy=policy,
                    static_masks=static_masks, collect=collect, sample_stride=stride)
    with context as selection:
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start+batch_size]
            batch = tokens[ids].to(device)
            logits = model(batch[:, :-1], use_cache=False).logits
            sums[ids] = next_token_sums(logits, batch[:, 1:])
            coverage[ids] = 1
            del logits
        stats = reduce_stats(selection)
    if dist.is_initialized():
        dist.all_reduce(sums); dist.all_reduce(coverage)
    if not (coverage == 1).all() or not torch.isfinite(sums).all():
        raise ValueError("Incomplete, duplicated or nonfinite evaluation")
    per_document = (sums / (tokens.shape[1]-1)).cpu().tolist()
    nll = float(np.mean(per_document))
    return {"loss": nll, "ppl": math.exp(nll), "documents": len(tokens),
            "scored_tokens": len(tokens)*(tokens.shape[1]-1),
            "per_document_nll": per_document}, stats


@torch.inference_mode()
def calibrate(model, tokens, keep, batch_size, stride, rank, world):
    device = next(model.parameters()).device
    indices = list(range(rank, len(tokens), world))
    with GroupSelection(model, keep=keep, policy="dense", collect=True,
                        sample_stride=stride) as selection:
        for start in range(0, len(indices), batch_size):
            model.model(tokens[indices[start:start+batch_size], :-1].to(device), use_cache=False)
        stats = reduce_stats(selection)
    if len(stats) != model.config.num_hidden_layers:
        raise ValueError("Missing calibration layer")
    choices = torch.stack([stats[i]["candidate_error_sum"].argmin() for i in range(len(stats))])
    masks = subset_masks(model.config.num_key_value_heads, keep, device=device)[choices]
    return masks, choices, stats


def report_stats(stats, choices):
    result = []
    for i, row in sorted(stats.items()):
        totals = row["totals"].cpu().tolist()
        raw, projected, joint, energy, count = totals
        static = row["candidate_error_sum"][choices[i]].item()
        if count <= 0 or energy <= 0 or not all(math.isfinite(x) for x in (*totals, static)):
            raise ValueError("Invalid dense-reference reconstruction statistics")
        result.append({"layer": i, "positions": int(count), "dense_output_energy_sum": energy,
            "error_over_energy": {"static": static/energy, "raw_norm": raw/energy,
                                  "projected_norm": projected/energy, "joint": joint/energy}})
    return result


def load_probe(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "passed" or sha256(path / "tokens.pt") != manifest["tokens_sha256"]:
        raise ValueError("Unverified probe")
    tokens = torch.load(path / "tokens.pt", map_location="cpu", weights_only=True)
    for split in ("calibration", "evaluation"):
        if tokens[split].shape != (manifest[split+"_documents"], manifest["length"]+1):
            raise ValueError("Probe shape differs from manifest")
        if tokens[split].dtype != torch.long:
            raise ValueError("Expected integer token IDs")
    return manifest, tokens


def load_model(path, device, dtype):
    return AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
        dtype=dtype, attn_implementation="sdpa").to(device).eval().requires_grad_(False)


def run(args):
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if args.device == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        if args.output.exists():
            raise ValueError("Refuse existing diagnostic report")
        manifest, tokens = load_probe(args.probe)
        if min(len(t) for t in tokens.values()) < world or args.stride > manifest["length"]:
            raise ValueError("Insufficient documents/positions for all ranks")
        # Verify the previously frozen model on rank zero before any forward pass.
        if rank == 0:
            for name, expected in manifest["model_files"].items():
                if sha256(Path(args.model) / name) != expected:
                    raise ValueError("Model changed after preparing probe: " + name)
        if world > 1:
            dist.barrier()
        torch.manual_seed(42)
        torch.set_float32_matmul_precision("highest")
        model = load_model(args.model, device, getattr(torch, args.dtype))
        if any(not 1 <= keep <= model.config.num_key_value_heads for keep in args.keep):
            raise ValueError("Invalid keep budget")
        report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": "joint_groups_dense_reference_v1", "backend": "sdpa", "dtype": args.dtype,
            "world_size": world, "batch_size_per_rank": args.batch_size, "sample_stride": args.stride,
            "model": str(args.model), "model_files": manifest["model_files"],
            "probe_manifest_sha256": sha256(args.probe / "manifest.json"),
            "probe": {k: manifest[k] for k in ("length", "seed", "calibration_documents", "evaluation_documents")},
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "datasets", "accelerate")},
            "limitations": ["Frozen pretrained model; no learned router", "Dense attention with masked outputs; no speedup measurement",
                             "Norm and joint policies use post-attention information", "Single prefixes of long English documents; not full validation PPL"],
            "budgets": {}}
        baseline = None
        for keep in args.keep:
            if rank == 0:
                print(f"CALIBRATE keep={keep}", flush=True)
            masks, choices, calibration_stats = calibrate(model, tokens["calibration"], keep,
                args.batch_size, args.stride, rank, world)
            budget = {"static_groups_per_layer": [m.nonzero().flatten().tolist() for m in masks],
                      "calibration_reconstruction": report_stats(calibration_stats, choices), "methods": {}}
            for policy in ("dense", "static", "raw_norm", "projected_norm", "joint"):
                if world > 1:
                    dist.barrier()
                start = time.monotonic()
                metrics, stats = evaluate(model, tokens["evaluation"], keep=keep, policy=policy,
                    static_masks=masks, collect=policy == "dense", batch_size=args.batch_size,
                    stride=args.stride, rank=rank, world=world)
                metrics["diagnostic_elapsed_seconds"] = time.monotonic()-start
                budget["methods"][policy] = metrics
                if policy == "dense":
                    budget["heldout_dense_reconstruction"] = report_stats(stats, choices)
                    if baseline is not None:
                        np.testing.assert_allclose(metrics["per_document_nll"], baseline, rtol=0, atol=1e-6)
                    baseline = metrics["per_document_nll"]
                if rank == 0:
                    print("METHOD_RESULT " + json.dumps({"keep": keep, "policy": policy,
                        **{k: v for k, v in metrics.items() if k != "per_document_nll"}}), flush=True)
            methods = budget["methods"]
            budget["joint_comparisons"] = {p: paired_summary(methods["joint"]["per_document_nll"],
                methods[p]["per_document_nll"]) for p in ("dense", "static", "raw_norm", "projected_norm")}
            report["budgets"][str(keep)] = budget
            if rank == 0:
                write(args.output.with_suffix(".partial.json"), report)
        report.update(status="passed", finished_utc=datetime.now(timezone.utc).isoformat())
        if rank == 0:
            write(args.output, report)
            print("DIAGNOSTIC_COMPLETE " + str(args.output), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--model", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--contract", type=Path, required=True)
    prep.add_argument("--length", type=int, default=2048)
    prep.add_argument("--calibration", type=int, default=64)
    prep.add_argument("--evaluation", type=int, default=128)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--max-shards", type=int, default=8)
    prep.add_argument("--exclude-manifest", type=Path)
    prep.add_argument("--recovery-from", type=Path)
    study = commands.add_parser("run")
    study.add_argument("--model", type=Path, required=True)
    study.add_argument("--probe", type=Path, required=True)
    study.add_argument("--output", type=Path, required=True)
    study.add_argument("--keep", type=int, nargs="+", default=[6, 4])
    study.add_argument("--batch-size", type=int, default=2)
    study.add_argument("--stride", type=int, default=16)
    study.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    study.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    if args.command == "prepare":
        receipt = prepare_probe(args.data_root, args.model, args.output, contract_path=args.contract,
            length=args.length, calibration=args.calibration, evaluation=args.evaluation,
            seed=args.seed, max_shards=args.max_shards, exclude_manifest=args.exclude_manifest,
            recovery_from=args.recovery_from)
        print("PROBE_PREPARED " + json.dumps({k: receipt[k] for k in ("status", "length", "calibration_documents", "evaluation_documents")}), flush=True)
    else:
        if min(args.batch_size, args.stride) < 1 or len(set(args.keep)) != len(args.keep):
            raise ValueError("Require positive sizes and distinct budgets")
        run(args)


if __name__ == "__main__":
    main()
