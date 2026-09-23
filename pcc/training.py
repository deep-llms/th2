"""Full-probe adapter training and frozen-teacher calibration, all local."""
import math
import time

import torch
from torch.nn import functional as F

from .model import Context
from .objectives import learning_rate
from .permutation import permute_targets


def frozen_teacher(teacher):
    if teacher.training or any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
        raise ValueError("Teacher must be in eval mode, frozen, with cleared gradients")


def prefix_batches(data, token_budget, microbatch, device):
    """Exact ordered input-token prefix, including a masked partial final context."""
    if microbatch < 1 or not 0 < token_budget <= int(data.valid.sum()):
        raise ValueError("Invalid prefix token budget or microbatch")
    remaining = token_budget
    for start in range(0, len(data), microbatch):
        context = data.batch(start, min(start + microbatch, len(data)), device)
        count = int(context.valid.sum())
        if count > remaining:
            ordinal = context.valid.flatten().long().cumsum(0).view_as(context.valid)
            valid = context.valid & (ordinal <= remaining)
            rows = int(valid.any(-1).sum())
            context = Context(context.input_ids[:rows], valid[:rows], context.position_ids[:rows],
                              context.segments[:rows] if context.segments is not None else None)
            count = remaining
        yield context
        remaining -= count
        if remaining == 0:
            break


@torch.no_grad()
def calibrate_scale(backbone, teacher, train, s, d, token_budget, microbatch):
    frozen_teacher(teacher)
    device = next(backbone.model.parameters()).device
    squared_sum, elements, input_tokens, target_tokens = 0., 0, 0, 0
    started = time.monotonic()
    for context in prefix_batches(train, token_budget, microbatch, device):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=backbone.model.dtype == torch.bfloat16):
            clean = backbone.clean(context, (s, d))
            delta = teacher(clean.states[s], clean.states[d], context.allowed(), clean.rotary)
        values = delta[:, :-1][context.targets()].double()
        squared_sum += float(values.square().sum())
        elements += values.numel()
        input_tokens += int(context.valid.sum())
        target_tokens += int(context.targets().sum())
    scale = math.sqrt(squared_sum / elements) if elements else float("nan")
    if input_tokens != token_budget or not math.isfinite(scale) or scale <= 1e-8:
        raise ValueError("No usable frozen teacher correction scale or incorrect calibration budget")
    return {"sigma_delta": scale, "squared_sum_fp64": squared_sum, "elements": elements,
            "input_tokens": input_tokens, "target_tokens": target_tokens,
            "elapsed_seconds": time.monotonic() - started}


def student_update(backbone, teacher, train, adapters, opts, s, d, start, per_update,
                   microbatch, update, scale, *, total_updates=610, warmup=31, permuted=False):
    """Train matched shallow arms; teacher correction exists only as supervision.

    A transient CPU pool spans this global update for permutation independent
    of microbatch boundaries. It is discarded after the update, never persisted
    as model memory. Extra clean forwards used to build it count in run timing.
    """
    frozen_teacher(teacher)
    allowed_arms = {"Target-Permuted"} if permuted else {"Shallow-ExtraAttn", "Student-PCC"}
    if set(adapters) != allowed_arms or set(opts) != allowed_arms:
        raise ValueError("Unexpected student training arms")
    if microbatch < 1 or per_update % microbatch or not math.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Invalid microbatch or correction scale")
    device = next(backbone.model.parameters()).device
    bf16 = backbone.model.dtype == torch.bfloat16
    global_context = train.batch(start, start + per_update, "cpu")
    target_count = int(global_context.targets().sum())
    if target_count <= 0:
        raise ValueError("No eligible targets in global update")
    pool = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
        for offset in range(start, start + per_update, microbatch):
            context = train.batch(offset, offset + microbatch, device)
            clean = backbone.clean(context, (s, d))
            correction = teacher(clean.states[s], clean.states[d], context.allowed(), clean.rotary)
            pool.append(correction.detach().cpu())
            del clean, correction
    targets = torch.cat(pool)
    permutation = None
    if permuted:
        targets, permutation = permute_targets(targets, global_context, update)
    for opt in opts.values():
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:
            group["lr"] = learning_rate(update, total=total_updates, warmup=warmup)
    totals = {name: {"lm_sum": 0., "alignment_sum": 0.} for name in adapters}
    for offset in range(start, start + per_update, microbatch):
        context = train.batch(offset, offset + microbatch, device)
        eligible = context.targets()
        target = targets[offset - start:offset - start + microbatch].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            clean = backbone.clean(context, (s,))
        for name, adapter in adapters.items():
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                # Only shallow states are ever inputs to a student/control.
                prediction = adapter(clean.states[s], clean.states[s], context.allowed(), clean.rotary)
                hidden = backbone.tail(clean, s, prediction, checkpoint_layers=True)
                sums, _ = backbone.losses(hidden, context)
                corr_sum = torch.zeros((), device=device)
                if name != "Shallow-ExtraAttn":
                    corr_sum = F.smooth_l1_loss(prediction[:, :-1][eligible].float() / scale,
                        target[:, :-1][eligible].detach().float() / scale, beta=1., reduction="sum")
                # Global normalization for both objectives; not a mean of means.
                loss = sums.sum() / target_count + corr_sum / (target_count * prediction.shape[-1])
            loss.backward()
            totals[name]["lm_sum"] += float(sums.detach().sum())
            totals[name]["alignment_sum"] += float(corr_sum.detach())
            del prediction, hidden, sums, corr_sum, loss
        del clean, target
    records = []
    for name, adapter in adapters.items():
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
        opts[name].step()
        records.append({"arm": name, "update": update,
            "input_tokens": update * int(global_context.valid.sum()), "target_tokens": target_count,
            "lm_loss": totals[name]["lm_sum"] / target_count,
            "correction_loss": totals[name]["alignment_sum"] / (target_count * backbone.model.config.hidden_size),
            "lambda_corr": 0. if name == "Shallow-ExtraAttn" else 1., "sigma_delta": scale,
            "lr": learning_rate(update, total=total_updates, warmup=warmup), "grad_norm": float(norm)})
    return records, permutation
