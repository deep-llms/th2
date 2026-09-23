"""Matched per-context NLL and correction diagnostics for the full probe."""
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from .screen import write_json
from .training import frozen_teacher


def summary(chunks):
    values = np.concatenate(chunks)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Empty/nonfinite diagnostic values")
    return {"count": len(values), "mean": float(values.mean(dtype=np.float64)),
            "median": float(np.median(values)), "p90": float(np.quantile(values, 0.9))}


@torch.no_grad()
def evaluate(backbone, teacher, students, data, s, d, output, *, microbatch=1, include_reference=True):
    frozen_teacher(teacher)
    if not set(students) <= {"Shallow-ExtraAttn", "Student-PCC", "Target-Permuted"}:
        raise ValueError("Unknown student evaluation arm")
    for adapter in students.values():
        if adapter.training or any(p.requires_grad for p in adapter.parameters()):
            raise ValueError("Freeze all evaluated adapters")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    names = (["Base", "Privileged-Deep"] if include_reference else []) + list(students)
    losses = {name: [] for name in names}
    diagnostics = {name: {"norm": [], "gate": [], "cosine_to_teacher": []} for name in names if name != "Base"}
    counts = []
    device = next(backbone.model.parameters()).device
    started = time.monotonic()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=backbone.model.dtype == torch.bfloat16):
        for start in range(0, len(data), microbatch):
            context = data.batch(start, min(start + microbatch, len(data)), device)
            eligible = context.targets()
            counts.extend(eligible.sum(-1).cpu().tolist())
            clean = backbone.clean(context, (s, d))
            target, teacher_diag = teacher(clean.states[s], clean.states[d], context.allowed(), clean.rotary, diagnostics=True)
            if include_reference:
                losses["Base"].extend(backbone.losses(clean.final_hidden, context)[0].double().cpu().tolist())
                hidden = backbone.tail(clean, s, target)
                losses["Privileged-Deep"].extend(backbone.losses(hidden, context)[0].double().cpu().tolist())
            for name in names:
                if name == "Base":
                    continue
                if name == "Privileged-Deep":
                    delta, diag = target, teacher_diag
                else:
                    # Actual one-pass evaluation; no deep tensor goes to the student.
                    hidden, delta, diag = backbone.student(context, s, students[name])
                    losses[name].extend(backbone.losses(hidden, context)[0].double().cpu().tolist())
                selected = delta[:, :-1][eligible].float()
                truth = target[:, :-1][eligible].float()
                diagnostics[name]["norm"].append(selected.norm(dim=-1).cpu().numpy())
                diagnostics[name]["gate"].append(diag["gate"][:, :-1, 0][eligible].float().cpu().numpy())
                diagnostics[name]["cosine_to_teacher"].append(F.cosine_similarity(selected, truth, dim=-1, eps=1e-8).cpu().numpy())
    for name, values in losses.items():
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite evaluation loss")
        np.savez(output / f"{name}.npz", loss_sums=np.array(values), target_counts=np.array(counts),
                 sequence_indices=np.arange(len(counts)))
    write_json(output / "diagnostics.json", {name: {key: summary(chunks) for key, chunks in metrics.items()}
                                            for name, metrics in diagnostics.items()})
    elapsed = time.monotonic() - started
    write_json(output / "timing.json", {"elapsed_seconds": elapsed,
        "arm_input_tokens_per_second": len(names) * int(data.valid.sum()) / elapsed,
        "teacher_diagnostic_cost_included": True,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
    return losses, counts
