"""Fixed-budget DDP optimization and resumable checkpoints for the four arms."""
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pcc.joint_training import rng_state, restore_rng
from .config import NAMES


def distributed():
    return dist.is_available() and dist.is_initialized()


def topology():
    return (dist.get_world_size(), dist.get_rank()) if distributed() else (1, 0)


def autocast(device, enabled):
    return torch.autocast(device.type, dtype=torch.bfloat16) if enabled else nullcontext()


def parameter_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def optimizer_for(model, recipe):
    groups = []
    for decay in (True, False):
        params = [p for p in model.parameters() if p.requires_grad and (p.ndim >= 2) == decay]
        if params:
            groups.append({"params": params, "weight_decay": recipe.weight_decay if decay else 0.0})
    return torch.optim.AdamW(groups, lr=recipe.learning_rate, betas=(0.9, 0.95), eps=1e-8, foreach=False)


def set_lr(optimizer, update, recipe):
    if not 1 <= update <= recipe.updates:
        raise ValueError("Update outside fixed training schedule")
    factor = update / recipe.warmup if update <= recipe.warmup else (
        0.1 + 0.9 * (1 + math.cos(math.pi * (update - recipe.warmup) /
                                 (recipe.updates - recipe.warmup))) / 2)
    for group in optimizer.param_groups:
        group["lr"] = recipe.learning_rate * factor


def train_update(model, parallel, optimizer, data, update, recipe, microbatch, mixed_precision):
    world, rank = topology()
    rows = recipe.tokens_per_update // recipe.context
    if rows % world or (rows // world) % microbatch:
        raise ValueError("Global batch must divide evenly across ranks and microbatches")
    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    set_lr(optimizer, update, recipe)
    first = (update - 1) * rows + rank * (rows // world)
    last = first + rows // world
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    start = time.monotonic()
    for offset in range(first, last, microbatch):
        context = data.batch(offset, offset + microbatch, device)
        if not bool(context.valid.all()) or context.segments is not None:
            raise ValueError("Pilot training requires full unpadded, full-causal contexts")
        sync = parallel.no_sync() if distributed() and offset + microbatch < last else nullcontext()
        with sync:
            with autocast(device, mixed_precision):
                result = parallel(context)
                loss = (result["lm_sum"] / (rows * (recipe.context - 1)) +
                        (result["k_sum"] + result["v_sum"]) / (2 * recipe.tokens_per_update)) * world
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite objective")
            loss.backward()
        totals += torch.stack([result[k].detach().double() for k in
                               ("lm_sum", "lm_count", "k_sum", "v_sum", "kv_count")])
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if distributed():
        dist.all_reduce(totals)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = torch.tensor(time.monotonic() - start, device=device, dtype=torch.float64)
    if distributed():
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    lm, targets, k, v, tokens = totals.tolist()
    if int(tokens) != recipe.tokens_per_update or int(targets) != rows * (recipe.context - 1):
        raise ValueError("Global token/target budget mismatch")
    return {"update": update, "input_tokens": update * recipe.tokens_per_update,
            "lm_loss": lm / targets, "loss_k": k / tokens, "loss_v": v / tokens,
            "objective": lm / targets + (k + v) / (2 * tokens),
            "grad_norm": float(norm), "lr": optimizer.param_groups[0]["lr"], "seconds": elapsed.item()}


@torch.no_grad()
def evaluate(model, data, rows, microbatch, mixed_precision):
    world, rank = topology()
    device = next(model.parameters()).device
    model.eval()
    if not 0 < rows <= len(data):
        raise ValueError("Evaluation row count outside prepared split")
    first, last = rows * rank // world, rows * (rank + 1) // world
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    for start in range(first, last, microbatch):
        with autocast(device, mixed_precision):
            result = model(data.batch(start, min(start + microbatch, last), device))
        totals += torch.stack([result[k].double() for k in ("lm_sum", "lm_count", "k_sum", "v_sum", "kv_count")])
    if distributed():
        dist.all_reduce(totals)
    lm, targets, k, v, tokens = totals.tolist()
    if targets <= 0 or tokens <= 0 or not bool(torch.isfinite(totals).all()):
        raise ValueError("Invalid evaluation totals")
    return {"lm_loss": lm / targets, "loss_k": k / tokens, "loss_v": v / tokens,
            "input_tokens": int(tokens), "target_tokens": int(targets), "rows": rows}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_checkpoint(path, model, optimizer, identity, update, history):
    world, rank = topology()
    local_rng = rng_state(current_device_only=True)
    states = [None] * world
    if distributed():
        dist.all_gather_object(states, local_rng)
    else:
        states[0] = local_rng
    if rank == 0:
        value = {"format": "deep-kv-checkpoint-v2", "identity": identity, "update": update,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "rng": states, "history": history}
        path = Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                torch.save(value, handle)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, path)
    if distributed():
        dist.barrier()


def load_checkpoint(path, model, optimizer, identity):
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    world, rank = topology()
    if state.get("format") != "deep-kv-checkpoint-v2" or state["identity"] != identity or len(state["rng"]) != world:
        raise ValueError("Checkpoint identity/topology mismatch")
    update = state["update"]
    if not 0 <= update <= identity["recipe"]["updates"]:
        raise ValueError("Invalid checkpoint update")
    if [row["update"] for row in state["history"]["train"]] != list(range(1, update + 1)):
        raise ValueError("Checkpoint training history is incomplete")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"][rank])
    return update, state["history"]


def train(model, train_data, eval_data, recipe, output, identity, *, microbatch=1,
          mixed_precision=True, resume=False, stop_after=None):
    recipe.validate()
    if identity.get("recipe") != asdict(recipe) or identity.get("arm") != model.arm:
        raise ValueError("Run identity differs from the requested arm/recipe")
    world, rank = topology()
    output = Path(output)
    device = next(model.parameters()).device
    # A cutoff preserves the full LR schedule and writes a resumable checkpoint;
    # it must never be accepted as an arm completion by the sequential runner.
    end = recipe.updates if stop_after is None else stop_after
    if not 1 <= end <= recipe.updates:
        raise ValueError("Invalid fixed-schedule cutoff")
    if rank == 0:
        if resume:
            if not (output / "checkpoint.pt").is_file() or (output / "complete.json").exists():
                raise ValueError("Resume requires an incomplete run with a checkpoint")
        else:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "run.json", identity)
    if distributed():
        dist.barrier()
    optimizer = optimizer_for(model, recipe)
    parallel = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                   broadcast_buffers=False) if distributed() else model
    update, history = 0, {"train": [], "evaluation": []}
    if resume:
        update, history = load_checkpoint(output / "checkpoint.pt", model, optimizer, identity)
        if rank == 0:
            if json.loads((output / "run.json").read_text()) != identity:
                raise ValueError("Run receipt differs from the checkpoint identity")
    if end < update:
        raise ValueError("Cutoff precedes checkpoint")
    if resume and rank == 0:
        (output / "stopped.json").unlink(missing_ok=True)
    if not history["evaluation"]:
        initial = evaluate(model, eval_data, recipe.monitor_rows, microbatch, mixed_precision)
        history["evaluation"].append({"update": 0, **initial})
    for update in range(update + 1, end + 1):
        row = train_update(model, parallel, optimizer, train_data, update, recipe, microbatch, mixed_precision)
        history["train"].append(row)
        if rank == 0:
            print(json.dumps({"arm": model.arm, **row}), flush=True)
        if update % recipe.eval_every == 0 or update == end:
            rows = recipe.eval_rows if update == recipe.updates else recipe.monitor_rows
            metrics = evaluate(model, eval_data, rows, microbatch, mixed_precision)
            history["evaluation"].append({"update": update, **metrics})
            if rank == 0:
                print(json.dumps({"evaluation": history["evaluation"][-1]}), flush=True)
        if update % recipe.checkpoint_every == 0 or update == end:
            save_checkpoint(output / "checkpoint.pt", model, optimizer, identity, update, history)
            if rank == 0 and update < end:
                write_json(output / "metrics.json", history)
    # Also handle interruption after the final checkpoint but before publication.
    if rank == 0:
        # A crash may leave a valid final checkpoint without metrics.json (or
        # with metrics from an older checkpoint). Repair it before completion.
        write_json(output / "metrics.json", history)
        status = {"status": "complete" if update == recipe.updates else "stopped",
                  "arm": model.arm, "name": NAMES[model.arm], "update": update,
                  "input_tokens": update * recipe.tokens_per_update,
                  "final_evaluation": history["evaluation"][-1],
                  "utc": datetime.now(timezone.utc).isoformat()}
        write_json(output / ("complete.json" if status["status"] == "complete" else "stopped.json"), status)
    if distributed():
        dist.barrier()
    return history
