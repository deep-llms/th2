"""The locked four-pair screen, single device with gradient accumulation.

Only adapters are optimized. There is intentionally no test-data argument and
no automatic full probe, student training, deployment, or pretraining launch.
"""
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .data import PreparedContexts, validate_matched_data
from .diagnostics import paired_adapters, run_preflight
from .objectives import optimizer, learning_rate
from .protocol import (CONTEXT, PAIRS, SCREEN_UPDATES, SCREEN_DEV_TOKENS,
                       TOKENS_PER_UPDATE, SCREEN_SEED)
from .statistics import select_pair


def write_json(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".pcc-", suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic publication without replacing a prior result.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def train_update(backbone, train, adapters, opts, s, d, start, per_update, microbatch, update, *, checkpoint_layers=True,
                 total_updates=128, warmup=7):
    """One global update, normalized by targets across all its microbatches.

    Production calls use a bf16 backbone. FP32 is supported here solely for
    independent numerical tests of accumulation and normalization.
    """
    if microbatch < 1 or per_update % microbatch:
        raise ValueError("Microbatch must divide the global context batch")
    device = next(backbone.model.parameters()).device
    bf16 = backbone.model.dtype == torch.bfloat16
    update_context = train.batch(start, start + per_update, "cpu")
    input_tokens = int(update_context.valid.sum())
    targets = int(update_context.targets().sum())
    if targets <= 0:
        raise ValueError("An optimizer update has no eligible causal targets")
    for opt in opts.values():
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:
            group["lr"] = learning_rate(update, total=total_updates, warmup=warmup)
    nll_sums = dict.fromkeys(adapters, 0.0)
    for offset in range(start, start + per_update, microbatch):
        context = train.batch(offset, offset + microbatch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            clean = backbone.clean(context, (s, d))
        for name, (adapter, source) in adapters.items():
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                delta = adapter(clean.states[s], clean.states[source], context.allowed(), clean.rotary)
                hidden = backbone.tail(clean, s, delta, checkpoint_layers=checkpoint_layers)
                sums, _ = backbone.losses(hidden, context)
                loss = sums.sum() / targets
            loss.backward()
            nll_sums[name] += float(sums.detach().sum())
            del delta, hidden, sums, loss
        del clean
    records = []
    for name, (adapter, _) in adapters.items():
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
        opts[name].step()
        records.append({"arm": name, "update": update, "input_tokens": update * input_tokens,
                        "target_tokens": targets, "nll": nll_sums[name] / targets,
                        "lr": learning_rate(update, total=total_updates, warmup=warmup), "grad_norm": float(norm)})
    return records


def screen(backbone, train_path, dev_path, output, *, microbatch=1, checkpoint_layers=True, provenance=None, load_data=None):
    per_update = TOKENS_PER_UPDATE // CONTEXT
    if microbatch < 1 or per_update % microbatch:
        raise ValueError("Microbatch must be a positive divisor of 16 contexts/update")
    load_data = load_data or PreparedContexts
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = backbone.model
    device = next(model.parameters()).device
    try:
        train = load_data(train_path, "train", SCREEN_UPDATES * TOKENS_PER_UPDATE)
        dev = load_data(dev_path, "dev", SCREEN_DEV_TOKENS)
        validate_matched_data(train, dev)
        if max(train.ids.max(), dev.ids.max()) >= backbone.model.config.vocab_size:
            raise ValueError("Prepared input IDs exceed the pinned vocabulary")
        write_json(output / "provenance.json", provenance if provenance is not None else {
            "model_config": model.config.to_dict(), "torch": torch.__version__,
            "model_revision": None, "invoked_via_python_api": True,
        })
        if model.dtype != torch.bfloat16:
            raise ValueError("The real screen requires a bf16 frozen backbone")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        preflight = run_preflight(backbone)
        write_json(output / "preflight.json", preflight)
        write_json(output / "data.json", {"train": train.metadata, "dev": dev.metadata,
                                         "train_path": str(train.path), "dev_path": str(dev.path)})
        losses, all_counts = {}, None
        started = time.monotonic()
        for s, d in PAIRS:
            deep, shallow = paired_adapters(backbone)
            adapters = {f"deep-{s}-{d}": (deep, d), f"shallow-{s}-{d}": (shallow, s)}
            opts = {name: optimizer(adapter) for name, (adapter, _) in adapters.items()}
            train_started = time.monotonic()
            with (output / f"train-{s}-{d}.jsonl").open("x") as log:
                for update in range(1, SCREEN_UPDATES + 1):
                    start = (update - 1) * per_update
                    records = train_update(backbone, train, adapters, opts, s, d, start,
                                           per_update, microbatch, update, checkpoint_layers=checkpoint_layers)
                    for record in records:
                        log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()
                    print(f"pair=({s},{d}) update={update}/{SCREEN_UPDATES}", flush=True)
            train_seconds = time.monotonic() - train_started
            for name, (adapter, _) in adapters.items():
                torch.save({"state_dict": adapter.state_dict(), "s": s, "d": d, "arm": name,
                            "init_seed": SCREEN_SEED, "updates": SCREEN_UPDATES,
                            "input_tokens": SCREEN_UPDATES * TOKENS_PER_UPDATE}, output / f"{name}.pt")
                adapter.eval()
            pair_losses = {name: [] for name in adapters}
            need_base = "Base" not in losses
            if need_base:
                pair_losses["Base"] = []
            diagnostics = {name: {"norms": [], "gates": []} for name in adapters}
            counts = []
            eval_started = time.monotonic()
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
                for start in range(0, len(dev), microbatch):
                    context = dev.batch(start, min(start + microbatch, len(dev)), device)
                    clean = backbone.clean(context, (s, d))
                    counts.extend(context.targets().sum(-1).cpu().tolist())
                    if need_base:
                        sums, _ = backbone.losses(clean.final_hidden, context)
                        pair_losses["Base"].extend(sums.double().cpu().tolist())
                    for name, (adapter, source) in adapters.items():
                        delta, diag = adapter(clean.states[s], clean.states[source], context.allowed(), clean.rotary,
                                              diagnostics=True)
                        hidden = backbone.tail(clean, s, delta)
                        sums, _ = backbone.losses(hidden, context)
                        pair_losses[name].extend(sums.double().cpu().tolist())
                        eligible = context.targets()
                        diagnostics[name]["norms"].append(delta[:, :-1].float().norm(dim=-1)[eligible].cpu().numpy())
                        diagnostics[name]["gates"].append(diag["gate"][:, :-1, 0].float()[eligible].cpu().numpy())
                        del delta, diag, hidden, sums
            if all_counts is not None and counts != all_counts:
                raise ValueError("Evaluation sequence counts changed across pairs")
            all_counts = counts
            losses.update(pair_losses)
            for name, values in pair_losses.items():
                np.savez(output / f"eval-{name}.npz", loss_sums=np.array(values),
                         target_counts=np.array(counts), sequence_indices=np.arange(len(counts)))
            def summary(chunks):
                values = np.concatenate(chunks)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite correction/gate diagnostics")
                return {"count": len(values), "mean": float(values.mean(dtype=np.float64)),
                        "median": float(np.median(values)), "p90": float(np.quantile(values, 0.9))}
            write_json(output / f"diagnostics-{s}-{d}.json", {
                name: {key: summary(chunks) for key, chunks in values.items()}
                for name, values in diagnostics.items()
            })
            eval_seconds = time.monotonic() - eval_started
            write_json(output / f"timing-{s}-{d}.json", {
                "paired_train_seconds": train_seconds,
                "paired_input_tokens_per_second": 2 * SCREEN_UPDATES * TOKENS_PER_UPDATE / train_seconds,
                "eval_seconds": eval_seconds,
                "evaluated_arm_input_tokens_per_second": len(pair_losses) * SCREEN_DEV_TOKENS / eval_seconds,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "clean_pass_cost_included": True,
            })
            del deep, shallow, adapters, opts
        decision = select_pair(losses, all_counts)
        decision["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "decision.json", decision)
        write_json(output / "complete.json", {"status": "ok", "decision": decision["decision"]})
        return decision
    except BaseException as error:
        try:
            write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        except Exception as report_error:
            error.add_note(f"Could not save failure.json: {report_error}")
        raise
