"""Phase-A route diagnostics and paired interventions, sharing packed LM scoring.

Invoked by packed_ppl.py --route_study. This module also compares two exported
samples without loading either model: python -m eval.route_study A.npz B.npz out.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from packed_evaluation import evaluate_packed, preserve_rng_and_mode


def _validate_routes(routes):
    routes = np.asarray(routes)
    if routes.ndim != 3 or not all(routes.shape) or routes.dtype.kind not in "iu":
        raise ValueError("Routes must be a nonempty integer [sequence, token, step] array")
    if routes.min() < 0 or routes.max() >= routes.shape[2]:
        raise ValueError("Phase-A module IDs must be in [0, S)")
    return routes


def summarize_routes(routes):
    """All input positions, including EOS/document starts, make routing decisions.

    Diversity uses complete, contiguous 10,000-token windows in validation order.
    Do not scale the global unique count by corpus size: uniqueness is nonlinear.
    """
    routes = _validate_routes(routes)
    steps = routes.shape[2]
    paths = np.ascontiguousarray(routes.reshape(-1, steps))
    # Opaque fixed-width keys preserve exact paths without base-S integer overflow.
    keys = paths.view(np.dtype((np.void, paths.dtype.itemsize * steps))).ravel()
    windows = len(paths) // 10_000
    diversity = [int(np.unique(keys[i*10_000:(i+1)*10_000]).size) for i in range(windows)]
    off = paths != np.arange(steps)
    repeated = (np.diff(np.sort(paths, axis=1), axis=1) == 0).any(axis=1)
    return dict(num_sequences=len(routes), input_tokens=len(paths), routing_decisions=paths.size,
        off_default_share=float(off.mean()), off_default_share_by_step=off.mean(0).tolist(),
        reuse_rate=float(repeated.mean()), distinct_paths=int(np.unique(keys).size),
        distinct_paths_per_10000_tokens=float(np.mean(diversity)) if windows else None,
        diversity_complete_windows=windows, diversity_excluded_tail_tokens=len(paths) % 10_000,
        mean_path_length=steps,
        module_counts_by_step=[np.bincount(paths[:, s], minlength=steps).tolist() for s in range(steps)])


class RouteCollector:
    """Collect only the diagnostic prefix from the evaluator's unpadded shards."""
    def __init__(self, accelerator, num_sequences, artifact_sequences):
        self.accelerator = accelerator
        self.num_sequences, self.artifact_sequences = num_sequences, artifact_sequences
        self.next_index = accelerator.process_index
        self.rows = []

    def __call__(self, batch, model):
        size = len(batch["input_ids"])
        indices = np.arange(size) * self.accelerator.num_processes + self.next_index
        self.next_index += size * self.accelerator.num_processes
        keep = int(np.count_nonzero(indices < self.num_sequences))
        if not keep:
            return
        if model.last_routes is None or tuple(model.last_routes.shape[1:]) != tuple(batch["input_ids"].shape):
            raise RuntimeError("Missing or misaligned captured routes")
        # Shape [sequence, token, step]; model uses [step, sequence, token].
        routes = model.last_routes[:, :keep].permute(1, 2, 0).to(device="cpu", dtype=torch.int32).numpy()
        for j in range(keep):
            index = int(indices[j])
            row = dict(index=index, routes=routes[j])
            if index < self.artifact_sequences:
                h = hashlib.sha256()
                for key in ("input_ids", "labels", "position_ids"):
                    a = batch[key][j].numpy().astype("<i8", copy=False)
                    h.update(key.encode() + a.tobytes())
                row.update(token_sha256=h.hexdigest(), input_ids=batch["input_ids"][j].numpy().copy())
            self.rows.append(row)

    def finish(self):
        accelerator = self.accelerator
        if accelerator.num_processes > 1:
            parts = [None] * accelerator.num_processes if accelerator.is_main_process else None
            torch.distributed.gather_object(self.rows, parts, dst=0)
            if not accelerator.is_main_process:
                return None
            rows = [row for part in parts for row in part]
        else:
            rows = self.rows
        rows.sort(key=lambda x: x["index"])
        if [row["index"] for row in rows] != list(range(self.num_sequences)):
            raise RuntimeError("Route study did not capture each diagnostic sequence exactly once")
        routes = np.stack([row["routes"] for row in rows])
        sample = rows[:self.artifact_sequences]
        return summarize_routes(routes), dict(
            routes=routes[:self.artifact_sequences],
            sequence_indices=np.arange(self.artifact_sequences, dtype=np.int64),
            input_ids=np.stack([row["input_ids"] for row in sample]),
            token_sha256=np.array([row["token_sha256"] for row in sample]))


def run_route_study(model, dataset, collator, accelerator, batch_size=1, *,
                    route_sequences=4096, artifact_sequences=256, routing_seed=0):
    """Three passes, one model/data load; no labels sent to the model's loss.

    Losses cover the entire selected held-out dataset, not just the route prefix.
    All ranks execute the same passes, using disjoint rows without sampler padding.
    The random intervention uses the existing rank/batch-indexed eval RNG, so its
    replay requires the same seed, batch size and world size (recorded by the CLI).
    """
    if not hasattr(model, "options") or model.options.mode != "compute_matched":
        raise ValueError("Route studies require a Phase-A B-DOT checkpoint")
    if not 1 <= artifact_sequences <= route_sequences <= len(dataset):
        raise ValueError("Require 1 <= route_artifact_sequences <= route_sequences <= held-out sequences; "
                         "set explicit smaller counts for a smoke test")
    collector = RouteCollector(accelerator, route_sequences, artifact_sequences)
    fields = ("routing_override", "capture_routes", "eval_seed", "eval_counter", "last_routes", "evaluation_policy")
    original = {key: getattr(model, key) for key in fields}
    metrics = {}
    with preserve_rng_and_mode(model):
        try:
            model.eval_seed = routing_seed
            model.evaluation_policy = 'target'
            for name, override in (("as_trained", None), ("fixed", "fixed"), ("random", "random")):
                if name != "as_trained" and (override == model.options.routing or model.options.r == 0):
                    metrics[name] = dict(metrics["as_trained"], reused_as_trained_pass=True)
                    continue
                model.routing_override = override
                model.capture_routes = True
                metrics[name] = evaluate_packed(model, dataset, collator, accelerator, batch_size,
                    batch_observer=collector if name == "as_trained" else None)
        finally:
            for key, value in original.items():
                setattr(model, key, value)
    reference = metrics["as_trained"]
    for name in ("fixed", "random"):
        if any(metrics[name][key] != reference[key] for key in ("num_tokens", "num_sequences")):
            raise RuntimeError("Route interventions scored different targets")
        metrics[name]["loss_delta_vs_as_trained"] = metrics[name]["loss"] - reference["loss"]
    collected = collector.finish()
    if not accelerator.is_main_process:
        return reference, None, None
    diagnostics, sample = collected
    report = dict(scope="Phase A: compute_matched, absolute windows, step K/V",
        routing_as_trained=model.options.routing, route_selection="first sequences in frozen validation order",
        diagnostic_sequences=route_sequences, artifact_sequences=artifact_sequences,
        route_diagnostics=diagnostics, interventions=metrics,
        routing_used_criterion_met=diagnostics["off_default_share"] >= 0.05,
        default_path_loss_increase=metrics["fixed"]["loss_delta_vs_as_trained"],
        default_path_criterion_met=metrics["fixed"]["loss_delta_vs_as_trained"] >= 0.01,
        cross_seed_agreement=None)
    return reference, report, sample


def write_route_sample(path, sample, metadata):
    """No pickle, no overwrite; publish only a complete compressed artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(handle, **sample,
                metadata_json=np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)))
            handle.flush()
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def compare_route_samples(first, second):
    """Module-ID agreement on identical exported sequences, not the full corpus.

    Different seeds cannot justify comparing different token identities, document
    boundaries, precision, backend or layouts. The caller supplies seed provenance;
    this function also permits a same-checkpoint repeatability check.
    """
    def read(path):
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    a, b = read(first), read(second)
    ma, mb = (json.loads(x["metadata_json"].item()) for x in (a, b))
    for key in ("format_version", "layout", "evaluation_contract"):
        if key not in ma or ma[key] != mb.get(key):
            raise ValueError(f"Route samples have incompatible {key}")
    for key in ("sequence_indices", "input_ids", "token_sha256"):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f"Route samples have different validation tokens/order/boundaries: {key}")
    ar, br = _validate_routes(a["routes"]), _validate_routes(b["routes"])
    if ar.shape != br.shape or ar.shape[:2] != a["input_ids"].shape:
        raise ValueError("Route sample shapes differ")
    equal = ar == br
    return dict(scope="exported route sample only", num_sequences=len(ar),
        input_tokens=int(np.prod(ar.shape[:2])), routing_decisions=ar.size,
        agreement=float(equal.mean()), agreement_by_step=equal.mean((0, 1)).tolist(),
        identical_path_fraction=float(equal.all(-1).mean()),
        first=dict(path=str(Path(first).resolve()), metadata=ma),
        second=dict(path=str(Path(second).resolve()), metadata=mb))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("output_file")
    args = parser.parse_args()
    from eval.packed_ppl import write_result
    write_result(args.output_file, compare_route_samples(args.first, args.second))
