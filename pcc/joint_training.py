"""Matched whole-model updates, fixed validation, atomic resumable checkpoints."""
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from .joint_config import JointSettings
from .screen import write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def autocast(model, enabled):
    device = next(model.parameters()).device
    return torch.autocast(device.type, dtype=torch.bfloat16) if enabled else nullcontext()


def make_optimizer(model):
    groups = {}
    for name, parameter in model.named_parameters():
        require(parameter.requires_grad and parameter.dtype == torch.float32,
                "All joint-training master parameters must be trainable fp32")
        branch = name.startswith("adapter.")
        decay = .01 if parameter.ndim >= 2 and name != "adapter.gate.weight" else 0.
        groups.setdefault((branch, decay), []).append(parameter)
    return torch.optim.AdamW([
        {"params": values, "lr": 3e-4 if branch else 1e-5,
         "peak_lr": 3e-4 if branch else 1e-5, "weight_decay": decay}
        for (branch, decay), values in groups.items()
    ], betas=(.9, .95), eps=1e-8, foreach=False)


def schedule(optimizer, update, settings):
    require(1 <= update <= settings.updates, "Update outside fixed budget")
    factor = update / settings.warmup if update <= settings.warmup else (
        .1 + .9 * (1 + math.cos(math.pi * (update - settings.warmup) /
                               (settings.updates - settings.warmup))) / 2)
    for group in optimizer.param_groups:
        group["lr"] = group["peak_lr"] * factor


def train_update(model, optimizer, data, update, settings, microbatch, *, mixed_precision=True, parallel=None):
    import torch.distributed as dist
    world, rank = (dist.get_world_size(), dist.get_rank()) if parallel is not None else (1, 0)
    per_update = settings.tokens_per_update // settings.context
    require(type(microbatch) is int and microbatch > 0 and per_update % microbatch == 0,
            "Microbatch must divide global context batch")
    require(per_update % world == 0 and (per_update // world) % microbatch == 0,
            "Global contexts must divide evenly across ranks and microbatches")
    start = (update - 1) * per_update
    global_context = data.batch(start, start + per_update, "cpu")
    require(int(global_context.valid.sum()) == settings.tokens_per_update, "Wrong global token count")
    targets = int(global_context.targets().sum())
    require(targets > 0, "No causal training targets")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    schedule(optimizer, update, settings)
    total = 0.
    started = time.monotonic()
    device = next(model.parameters()).device
    local_start = start + rank * (per_update // world)
    local_end = local_start + per_update // world
    for offset in range(local_start, local_end, microbatch):
        context = data.batch(offset, offset + microbatch, device)
        sync = parallel.no_sync() if parallel is not None and offset + microbatch < local_end else nullcontext()
        with sync:
            with autocast(model, mixed_precision):
                sums, _ = parallel(context) if parallel is not None else model.losses(model(context), context)
                # DDP averages gradients; compensate to preserve the exact
                # global target-normalized objective over all 16 contexts.
                loss = sums.sum() * world / targets
            require(bool(torch.isfinite(loss)), "Nonfinite training loss")
            loss.backward()
        total += float(sums.detach().sum())
        del context, sums, loss
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    if parallel is not None:
        value = torch.tensor(total, dtype=torch.float64, device=device)
        dist.all_reduce(value)
        total = value.item()
        value = torch.tensor(elapsed, dtype=torch.float64, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        elapsed = value.item()
    return {"update": update, "input_tokens": update * settings.tokens_per_update,
            "target_tokens": targets, "nll": total / targets, "grad_norm": float(norm),
            "learning_rates": [g["lr"] for g in optimizer.param_groups],
            "seconds": elapsed}


@torch.no_grad()
def evaluate(model, data, microbatch, *, mixed_precision=True, rows=None, distributed=False):
    import torch.distributed as dist
    world, rank = (dist.get_world_size(), dist.get_rank()) if distributed else (1, 0)
    was_training = model.training
    model.eval()
    sums, counts = [], []
    limit = len(data) if rows is None else rows
    require(0 < limit <= len(data), "Invalid evaluation slice")
    device = next(model.parameters()).device
    first, last = limit * rank // world, limit * (rank + 1) // world
    try:
        for start in range(first, last, microbatch):
            context = data.batch(start, min(start + microbatch, last), device)
            with autocast(model, mixed_precision):
                losses, targets = model.losses(model(context), context)
            sums.extend(losses.double().cpu().tolist())
            counts.extend(targets.cpu().tolist())
    finally:
        model.train(was_training)
    sums, counts = np.array(sums, dtype=np.float64), np.array(counts, dtype=np.int64)
    if distributed:
        # Disjoint contiguous shards, without DistributedSampler padding or
        # repeated validation rows. All ranks recover original sequence order.
        combined = torch.zeros((2, limit), dtype=torch.float64, device=device)
        combined[0, first:last] = torch.as_tensor(sums, device=device)
        combined[1, first:last] = torch.as_tensor(counts, device=device)
        dist.all_reduce(combined)
        sums = combined[0].cpu().numpy().copy()
        counts = combined[1].cpu().numpy().astype(np.int64)
    require(np.isfinite(sums).all() and (counts > 0).all(), "Invalid validation losses")
    return sums, counts


def rng_state(*, current_device_only=False):
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
            "torch": torch.get_rng_state(),
            "cuda": ([torch.cuda.get_rng_state()] if current_device_only else torch.cuda.get_rng_state_all())
                    if torch.cuda.is_initialized() else [],
            "cuda_scope": "current" if current_device_only else "all"}


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if state.get("cuda_scope") == "current":
            require(torch.cuda.is_available() and len(state["cuda"]) == 1, "Resume CUDA topology mismatch")
            torch.cuda.set_rng_state(state["cuda"][0])
        else:
            require(torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count(), "Resume CUDA topology mismatch")
            torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, identity, update, history, *, rank_rng=None):
    path = Path(path)
    state = {"format": "joint-v1", "identity": identity, "update": update,
             "next_context": update * identity["settings"]["tokens_per_update"] // identity["settings"]["context"],
             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "history": history, "rng": rank_rng[0] if rank_rng is not None else rng_state()}
    if rank_rng is not None:
        state["rank_rng"] = rank_rng
    # Replace only this experiment's latest checkpoint after a complete fsynced write.
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".joint-", suffix=".part", delete=False) as handle:
            name = Path(handle.name)
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def load_checkpoint(path, model, optimizer, identity, *, rng_rank=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(state.get("format") == "joint-v1" and state["identity"] == identity,
            "Resume configuration/data/code identity mismatch")
    update = state["update"]
    settings = identity["settings"]
    require(type(update) is int and 0 <= update <= settings["updates"], "Invalid checkpoint update")
    require(state["next_context"] == update * settings["tokens_per_update"] // settings["context"], "Invalid data cursor")
    require([r["update"] for r in state["history"]["train"]] == list(range(1, update + 1)), "Incomplete checkpoint training history")
    require(all(r["input_tokens"] == r["update"] * settings["tokens_per_update"] for r in state["history"]["train"]), "Wrong checkpoint token history")
    require(all(0 <= r["update"] <= update and math.isfinite(r["nll"]) for r in state["history"]["validation"]), "Invalid checkpoint validation history")
    require(all(bool(torch.isfinite(t).all()) for t in state["model"].values()), "Nonfinite checkpoint weights")
    expected_groups = optimizer.state_dict()["param_groups"]
    saved_groups = state["optimizer"]["param_groups"]
    require(len(saved_groups) == len(expected_groups), "Wrong optimizer group count")
    for saved, expected in zip(saved_groups, expected_groups):
        require(all(saved[k] == expected[k] for k in ("params", "peak_lr", "weight_decay", "betas", "eps")),
                "Wrong optimizer recipe in checkpoint")
    for moments in state["optimizer"]["state"].values():
        require(float(moments["step"]) == update, "Wrong optimizer step in checkpoint")
        require(all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in (moments["exp_avg"], moments["exp_avg_sq"])),
                "Invalid optimizer moments")
    # Reject inconsistent copies of a tied weight rather than silently loading
    # the last copy and changing both parameters.
    if model.model.config.tie_word_embeddings:
        require(torch.equal(state["model"]["model.model.embed_tokens.weight"], state["model"]["model.lm_head.weight"]), "Inconsistent tied checkpoint weights")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if rng_rank is not None:
        require(len(state.get("rank_rng", [])) == identity["distributed"]["world_size"], "Missing per-rank RNG states")
        restore_rng(state["rank_rng"][rng_rank])
    else:
        restore_rng(state["rng"])
    return update, state["history"]


def audit_final_checkpoint(path, identity):
    """Read checkpoint contents without constructing another full model."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    settings = identity["settings"]
    require(state.get("format") == "joint-v1" and state["identity"] == identity, "Final checkpoint identity mismatch")
    require(state["update"] == settings["updates"] and
            state["next_context"] == settings["updates"] * settings["tokens_per_update"] // settings["context"],
            "Final checkpoint budget mismatch")
    require([r["update"] for r in state["history"]["train"]] == list(range(1, settings["updates"] + 1)),
            "Incomplete final checkpoint history")
    require(bool(state["model"]) and bool(state["optimizer"]["state"]), "Empty final checkpoint")
    if "distributed" in identity:
        require(len(state.get("rank_rng", [])) == identity["distributed"]["world_size"],
                "Final checkpoint missing per-rank RNG")
    for tensor in state["model"].values():
        require(tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()), "Invalid final checkpoint tensor")
    for moments in state["optimizer"]["state"].values():
        require(float(moments["step"]) == settings["updates"], "Final optimizer step mismatch")
        require(all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in (moments["exp_avg"], moments["exp_avg_sq"])), "Invalid final optimizer moments")
    require(torch.equal(state["model"]["model.model.embed_tokens.weight"], state["model"]["model.lm_head.weight"]), "Final checkpoint lost tied weights")
    return {"status": "ok", "update": state["update"], "history": state["history"]}


def train(model, train_data, dev, output, identity, *, microbatch=1, resume=None,
          mixed_precision=True, stop_after=None, distributed=False):
    import torch.distributed as dist
    world, rank = (dist.get_world_size(), dist.get_rank()) if distributed else (1, 0)
    if distributed:
        require(identity.get("distributed", {}).get("world_size") == world,
                "Distributed identity/world size mismatch")
    settings = JointSettings(**identity["settings"]).validate()
    require(len(train_data) * settings.context == settings.updates * settings.tokens_per_update,
            "Training pool does not match complete fixed budget")
    require(int(dev.valid.sum()) == settings.dev_tokens, "Incorrect validation budget")
    require(identity["arm"] == model.arm, "Model arm differs from run identity")
    require((model.s, model.d) == (settings.s, settings.d), "Model coordinates differ from run identity")
    require(stop_after is None or type(stop_after) is int and 0 < stop_after <= settings.updates,
            "Invalid bounded stop")
    output = Path(output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    if distributed:
        dist.barrier()
    try:
        started = time.monotonic()
        if rank == 0:
            write_json(output / "identity.json", identity)
            write_json(output / "invocation.json", {"resume": str(Path(resume).resolve()) if resume else None,
                                                   "stop_after": stop_after})
        optimizer = make_optimizer(model)
        update, history = (0, {"train": [], "validation": []})
        if resume:
            update, history = load_checkpoint(resume, model, optimizer, identity,
                                               rng_rank=rank if distributed else None)
        parallel = None
        if distributed:
            from .distributed import wrap
            parallel = wrap(model)
        target = settings.updates if stop_after is None else stop_after
        require(update <= target, "Resume step exceeds requested endpoint")
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def monitor(step):
            sums, counts = evaluate(model, dev, microbatch, mixed_precision=mixed_precision,
                                    rows=settings.monitor_tokens // settings.context, distributed=distributed)
            history["validation"].append({"update": step, "nll": float(sums.sum() / counts.sum()),
                                          "target_tokens": int(counts.sum())})

        def checkpoint(step):
            rank_rng = None
            if distributed:
                rank_rng = [None] * world
                dist.all_gather_object(rank_rng, rng_state(current_device_only=True))
            if rank == 0:
                save_checkpoint(output / "latest.pt", model, optimizer, identity, step, history, rank_rng=rank_rng)
            if distributed:
                dist.barrier()

        if update == 0 and not history["validation"]:
            monitor(0)
        with (output / "train.jsonl" if rank == 0 else Path(os.devnull)).open("x" if rank == 0 else "w") as log, \
             (output / "validation.jsonl" if rank == 0 else Path(os.devnull)).open("x" if rank == 0 else "w") as validation_log:
            for r in history["train"]:
                log.write(json.dumps(r, allow_nan=False) + "\n")
            for r in history["validation"]:
                validation_log.write(json.dumps(r, allow_nan=False) + "\n")
            log.flush(); validation_log.flush()
            for step in range(update + 1, target + 1):
                record = train_update(model, optimizer, train_data, step, settings, microbatch,
                                      mixed_precision=mixed_precision, parallel=parallel)
                history["train"].append(record)
                log.write(json.dumps(record, allow_nan=False) + "\n"); log.flush()
                if step % settings.eval_every == 0 or step == target:
                    monitor(step)
                    validation_log.write(json.dumps(history["validation"][-1], allow_nan=False) + "\n")
                    validation_log.flush()
                    checkpoint(step)
                if rank == 0:
                    print(f"arm={model.arm} update={step}/{settings.updates} nll={record['nll']:.6f}", flush=True)
            if not (output / "latest.pt").exists():
                checkpoint(target)
        if target != settings.updates:
            report = {"status": "stopped_at_step", "update": target, "complete": False}
            if rank == 0:
                write_json(output / "stopped.json", report)
            return report
        # No second multi-GB copy: both names refer to the same final inode.
        if rank == 0:
            os.link(output / "latest.pt", output / "final.pt")
        sums, counts = evaluate(model, dev, microbatch, mixed_precision=mixed_precision, distributed=distributed)
        if rank == 0:
            np.savez(output / "eval.npz", loss_sums=sums, target_counts=counts,
                     sequence_indices=np.arange(len(counts)))
        report = {"status": "ok", "arm": model.arm, "updates": target,
                  "input_tokens": target * settings.tokens_per_update,
                  "nll": float(sums.sum() / counts.sum()), "target_tokens": int(counts.sum()),
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  "elapsed_seconds_this_invocation": time.monotonic() - started,
                  "training_seconds_all_updates": sum(r["seconds"] for r in history["train"])}
        if distributed:
            rank_resources = [None] * world
            dist.all_gather_object(rank_resources, {
                "rank": rank, "peak_allocated_bytes": report["cuda_peak_allocated_bytes"],
                "elapsed_seconds": report["elapsed_seconds_this_invocation"]})
            report["rank_resources"] = rank_resources
            report["world_size"] = world
        if rank == 0:
            audit_final_checkpoint(output / "final.pt", identity)
            report["checkpoint_verified"] = True
            write_json(output / "complete.json", report)
        if distributed:
            dist.barrier()
        return report
    except BaseException as error:
        if rank == 0:
            write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        raise


def code_identity():
    # Preserve enough evidence to refuse mixed-code resume even without Git.
    paths = sorted(Path(__file__).parent.glob("*.py"))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
