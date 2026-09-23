"""Paired sequence bootstrap; token-weighted estimators, fixed screen gates."""
import math
import numpy as np
from .protocol import DATA_SEED, PAIRS


def paired_bootstrap(loss_sums, counts, *, resamples=1000, seed=DATA_SEED):
    counts = np.asarray(counts)
    losses = {name: np.asarray(values, dtype=np.float64) for name, values in loss_sums.items()}
    if counts.ndim != 1 or len(counts) < 2 or not np.issubdtype(counts.dtype, np.integer) or np.any(counts <= 0):
        raise ValueError("Need at least two sequences with positive target counts")
    if type(resamples) is not int or resamples < 1 or not losses or any(
        x.shape != counts.shape or not np.isfinite(x).all() or (x < 0).any() for x in losses.values()
    ):
        raise ValueError("Invalid paired loss records or bootstrap budget")
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(resamples) for name in losses}
    for i in range(resamples):
        indices = rng.integers(0, len(counts), len(counts))
        denominator = counts[indices].sum()
        for name, values in losses.items():
            samples[name][i] = values[indices].sum() / denominator
    return {"nll": {name: float(x.sum() / counts.sum()) for name, x in losses.items()},
            "samples": samples}


def select_pair(losses, counts, *, resamples=1000):
    expected = {"Base"} | {f"{arm}-{s}-{d}" for s, d in PAIRS for arm in ("deep", "shallow")}
    if set(losses) != expected:
        raise ValueError("Screen requires Base and both arms for all four preregistered pairs")
    stats = paired_bootstrap(losses, counts, resamples=resamples)
    nll, samples = stats["nll"], stats["samples"]
    pairs = []
    for s, d in PAIRS:
        deep, shallow = f"deep-{s}-{d}", f"shallow-{s}-{d}"
        def contrast(a, b):
            lo, hi = np.quantile(samples[a] - samples[b], [0.025, 0.975])
            return {"difference": nll[a] - nll[b], "ci95": [float(lo), float(hi)]}
        base_contrast, source_contrast = contrast(deep, "Base"), contrast(deep, shallow)
        pairs.append({"s": s, "d": d, "deep_nll": nll[deep],
                      "deep_source_advantage": nll[shallow] - nll[deep],
                      "deep_minus_base": base_contrast, "deep_minus_shallow": source_contrast,
                      "eligible": base_contrast["ci95"][1] < 0 and source_contrast["ci95"][1] < 0})
    eligible = [p for p in pairs if p["eligible"]]
    selected = min(eligible, key=lambda p: (-p["deep_source_advantage"], p["deep_nll"], p["s"], p["d"])) if eligible else None
    return {"stage": "layer_pair_screen", "nll": nll,
            "perplexity": {name: math.exp(value) for name, value in nll.items()}, "pairs": pairs,
            "selected_pair": [selected["s"], selected["d"]] if selected else None,
            "decision": "proceed_to_full_probe" if selected else "stop_negative_screen",
            "bootstrap_resamples": resamples, "bootstrap_seed": DATA_SEED,
            "test_unlocked": False, "pretraining_authorized": False}


def probe_decision(losses, counts, *, split="dev"):
    required = {"Base", "Privileged-Deep", "Shallow-ExtraAttn", "Student-PCC"}
    if not required <= set(losses) or set(losses) - (required | {"Target-Permuted"}):
        raise ValueError("Missing or unknown full-probe arms")
    if split not in ("dev", "test"):
        raise ValueError("Expected dev or test")
    if split == "test" and "Target-Permuted" not in losses:
        raise ValueError("Confirmatory test requires all five frozen arms")
    resamples = 2000 if split == "dev" else 5000
    stats = paired_bootstrap(losses, counts, resamples=resamples)
    nll, samples = stats["nll"], stats["samples"]
    pairs = {"privileged": ("Privileged-Deep", "Base"),
             "deep_source": ("Privileged-Deep", "Shallow-ExtraAttn"),
             "pcc": ("Student-PCC", "Base"),
             "distillation": ("Student-PCC", "Shallow-ExtraAttn")}
    if "Target-Permuted" in losses:
        pairs["target_semantics"] = ("Student-PCC", "Target-Permuted")
    contrasts = {}
    for name, (a, b) in pairs.items():
        contrasts[name] = {"arms": [a, b], "difference": nll[a] - nll[b],
                           "ci95": np.quantile(samples[a] - samples[b], [0.025, 0.975]).tolist()}
    privileged_gain = nll["Base"] - nll["Privileged-Deep"]
    student_gain = nll["Base"] - nll["Student-PCC"]
    rho = student_gain / privileged_gain if privileged_gain > 0 else None
    gates = {name: contrast["ci95"][1] < 0 for name, contrast in contrasts.items()}
    gates["privileged"] = gates["privileged"] and privileged_gain >= 0.0005
    gates["recovery"] = rho is not None and rho >= 0.25
    conditional = gates["pcc"] and gates["distillation"]
    gates["target_semantics"] = gates.get("target_semantics", False) if conditional else True
    passed = all(gates.values())
    # The contract's locked-test criterion is directional consistency, with CIs
    # reported. Do not invent a post-test significance or recovery threshold.
    consistent = len(contrasts) == 5 and all(c["difference"] < 0 for c in contrasts.values())
    return {"stage": f"probe_{split}", "nll": nll,
            "perplexity": {name: math.exp(value) for name, value in nll.items()},
            "contrasts": contrasts, "privileged_gain": privileged_gain, "student_gain": student_gain,
            "recovery_ratio": rho, "gates": gates, "conditional_control_required": conditional,
            "all_dev_gates_pass": passed if split == "dev" else None,
            "directionally_consistent": consistent if split == "test" else None,
            "bootstrap_resamples": resamples, "bootstrap_seed": DATA_SEED,
            "test_unlocked": passed if split == "dev" else True, "pretraining_authorized": False}
