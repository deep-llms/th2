"""HF Trainer/Accelerate training with the project's loss and matched-run receipts."""
from dataclasses import asdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import SequentialSampler, Subset
from transformers import Trainer, TrainerCallback, TrainingArguments, default_data_collator
from transformers.trainer_callback import TrainerState

from pcc.model import Context
from .config import NAMES
from .data import sha256


def distributed():
    return dist.is_available() and dist.is_initialized()


def topology():
    return (dist.get_world_size(), dist.get_rank()) if distributed() else (1, 0)


def parameter_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def summarize_statistics(statistics):
    values = np.asarray(statistics, dtype=np.float64)
    lm, targets, k, v, tokens = values.sum(axis=0).tolist()
    if targets <= 0 or tokens <= 0 or not np.isfinite(values).all():
        raise ValueError("Nonfinite loss or empty token statistics")
    return {"lm_loss": lm / targets, "loss_k": k / tokens, "loss_v": v / tokens,
            "objective": lm / targets + (k + v) / (2 * tokens),
            "input_tokens": int(tokens), "target_tokens": int(targets), "rows": len(values)}


def compute_metrics(prediction):
    result = summarize_statistics(prediction.predictions)
    result["loss"] = result["objective"]
    return result


@contextmanager
def offline_wandb_run(args):
    """Own one offline run per arm, including repeated calls in one process."""
    if "wandb" not in args.report_to or args.process_index != 0:
        yield
        return
    import wandb
    if wandb.run is not None:
        raise ValueError("Deep-KV requires its own W&B run; finish the existing run first")
    directory = Path(os.environ.get("WANDB_DIR", args.output_dir)).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # An explicit dir avoids W&B's cached environment retaining a previous arm.
    # The standard HF callback attaches to this run; the context closes it even
    # if training fails. Nonzero ranks and normal CPU tests never initialize it.
    with wandb.init(mode="offline", project=os.environ.get("WANDB_PROJECT", "deep2shallow"),
                    name=args.run_name, dir=str(directory)):
        yield


def training_arguments(recipe, output, microbatch, *, cpu=False, mixed_precision=True):
    rows = recipe.tokens_per_update // recipe.context
    world = topology()[0]
    if type(microbatch) is not int or microbatch <= 0 or rows % (world * microbatch):
        raise ValueError("Global batch must divide evenly across ranks and microbatches")
    return TrainingArguments(
        output_dir=str(output), use_cpu=cpu, bf16=mixed_precision,
        per_device_train_batch_size=microbatch, per_device_eval_batch_size=microbatch,
        gradient_accumulation_steps=rows // (world * microbatch),
        num_train_epochs=1, max_steps=recipe.updates,
        learning_rate=recipe.learning_rate, lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1}, warmup_steps=recipe.warmup,
        weight_decay=recipe.weight_decay, adam_beta1=0.9, adam_beta2=0.95,
        adam_epsilon=1e-8, max_grad_norm=1.0,
        # Match baseline's Trainer default (fused AdamW on this CUDA runtime).
        optim="adamw_torch" if cpu else "adamw_torch_fused",
        seed=recipe.seed, data_seed=recipe.data_seed,
        logging_steps=recipe.logging_every, logging_nan_inf_filter=False,
        # Native rotation runs before our all-rank checkpoint certification and
        # counts partial saves. Retain two certified saves in on_save instead.
        save_steps=recipe.checkpoint_every, save_total_limit=None,
        eval_strategy="steps", eval_steps=recipe.eval_every, eval_on_start=True,
        dataloader_num_workers=recipe.dataloader_workers, dataloader_pin_memory=not cpu,
        ddp_timeout=21600, ddp_find_unused_parameters=False,
        # DeepKV already checkpoints its custom blocks and chunked output head.
        gradient_checkpointing=False, remove_unused_columns=False,
        report_to="none" if cpu else "wandb", run_name=f"deep-kv-{Path(output).name}-seed{recipe.seed}",
        disable_tqdm=True, push_to_hub=False,
    )


class DeepKVTrainer(Trainer):
    """Keep the HF loop; customize loss, fixed input order, and compact evaluation."""

    def __init__(self, *args, **kwargs):
        import accelerate
        import transformers
        if transformers.__version__ != "5.9.0" or accelerate.__version__ != "1.13.0":
            raise ValueError("Use transformers==5.9.0 and accelerate==1.13.0 for this Trainer integration")
        super().__init__(*args, **kwargs)
        # We return a microbatch mean, not a sum normalized by HF's label count.
        # Trainer divides once by accumulation; DDP averages once across ranks.
        self.model_accepts_loss_kwargs = False
        self.step_totals = None

    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ids = inputs["input_ids"]
        context = Context(ids, torch.ones_like(ids, dtype=torch.bool),
                          torch.arange(ids.shape[1], device=ids.device).expand_as(ids))
        outputs = model(context)
        loss = (outputs["lm_sum"] / outputs["lm_count"] +
                (outputs["k_sum"] + outputs["v_sum"]) / (2 * outputs["kv_count"]))
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite objective")
        if model.training and self.is_in_train:
            totals = outputs["statistics"].sum(dim=0)
            self.step_totals = totals if self.step_totals is None else self.step_totals + totals
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        stats = outputs["statistics"]
        # A per-example placeholder makes Trainer invoke compute_metrics. HF
        # gathers and truncates both tensors to the true eval dataset length.
        labels = torch.zeros(len(stats), dtype=torch.long, device=stats.device)
        # A batch-mean scalar cannot be de-duplicated correctly when Accelerate
        # pads an uneven eval shard. Compute eval_loss from the gathered rows.
        return None, stats, labels

    def _save(self, output_dir=None, state_dict=None):
        # nn.Module wrappers do not get PreTrainedModel's tied-weight handling.
        # Preserve BOTH names for strict restore, cloning only shared storage.
        state = dict(self.model.state_dict() if state_dict is None else state_dict)
        seen = set()
        for name, value in state.items():
            key = (value.device, value.untyped_storage().data_ptr())
            if key in seen:
                state[name] = value.clone()
            seen.add(key)
        super()._save(output_dir, state_dict=state)

    def _load_optimizer_and_scheduler(self, checkpoint):
        # Accelerate 1.13 can expose cpu:0 in multi-CPU runs. Trainer 5.9
        # passes that as torch.load's map_location, which this PyTorch rejects.
        # Normalize only the CPU restore path; CUDA uses native HF loading.
        if checkpoint is not None and self.args.device.type == "cpu":
            self.optimizer.load_state_dict(torch.load(Path(checkpoint) / "optimizer.pt",
                                                      map_location="cpu", weights_only=True))
            self.lr_scheduler.load_state_dict(torch.load(Path(checkpoint) / "scheduler.pt",
                                                         map_location="cpu", weights_only=True))
        else:
            super()._load_optimizer_and_scheduler(checkpoint)


def checkpoint_files(world):
    return ["model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json",
            "training_args.bin"] + (["rng_state.pth"] if world == 1 else
                                    [f"rng_state_{rank}.pth" for rank in range(world)])


def verify_checkpoint(path, identity, *, verify_hashes=True):
    path = Path(path)
    receipt = json.loads((path / "deep_kv.json").read_text())
    state = json.loads((path / "trainer_state.json").read_text())
    if (receipt.get("format") != "deep-kv-hf-checkpoint-v1" or receipt["identity"] != identity
            or state["global_step"] != receipt["update"]
            or state["max_steps"] != identity["recipe"]["updates"]
            or not 0 < receipt["update"] <= state["max_steps"]):
        raise ValueError("Checkpoint identity/schedule mismatch")
    if set(receipt["files"]) != set(checkpoint_files(identity.get("world_size", topology()[0]))):
        raise ValueError("Incomplete Trainer checkpoint")
    for name, spec in receipt["files"].items():
        file = path / name
        if file.stat().st_size != spec["bytes"] or (verify_hashes and sha256(file) != spec["sha256"]):
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    if [row["update"] for row in receipt["history"]["train"]] != list(range(1, receipt["update"] + 1)):
        raise ValueError("Checkpoint training history is incomplete")
    return receipt


class ExperimentCallback(TrainerCallback):
    def __init__(self, recipe, identity, end, evaluation, history):
        self.recipe, self.identity, self.end = recipe, identity, end
        self.evaluation, self.history = evaluation, history
        self.trainer = None
        self.lr = None

    def on_step_begin(self, args, state, control, optimizer, **kwargs):
        self.lr = optimizer.param_groups[0]["lr"]

    def on_step_end(self, args, state, control, **kwargs):
        totals = self.trainer.step_totals
        self.trainer.step_totals = None
        if totals is None:
            raise ValueError("Missing custom training statistics")
        if distributed():
            dist.all_reduce(totals)
        tokens = int(totals[-1].item())
        targets = int(totals[1].item())
        rows = self.recipe.tokens_per_update // self.recipe.context
        if tokens != self.recipe.tokens_per_update or targets != rows * (self.recipe.context - 1):
            raise ValueError("Global token/target budget mismatch")
        metrics = summarize_statistics(totals.cpu().numpy()[None])
        metrics.update(update=state.global_step, input_tokens=state.global_step * tokens,
                       rows=rows, lr=self.lr)
        self.history["train"].append(metrics)
        if state.global_step >= self.end:
            control.should_training_stop = True
            control.should_save = control.should_evaluate = True
            self.trainer.eval_dataset = self.evaluation
        return control

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        keys = ("lm_loss", "loss_k", "loss_v", "objective", "input_tokens", "target_tokens", "rows")
        self.history["evaluation"].append({"update": state.global_step,
                                            **{key: metrics[f"eval_{key}"] for key in keys}})

    def on_log(self, args, state, control, logs, **kwargs):
        if "loss" in logs and self.history["train"]:
            # HF logs interval-averaged total loss; these are explicitly the
            # most recent optimizer step's individually normalized components.
            row = self.history["train"][-1]
            logs.update({f"step_{key}": row[key] for key in ("lm_loss", "loss_k", "loss_v")})

    def on_save(self, args, state, control, **kwargs):
        # All ranks must finish RNG writes before rank zero certifies a save.
        self.trainer.accelerator.wait_for_everyone()
        if state.is_world_process_zero:
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            files = {name: {"bytes": (path / name).stat().st_size, "sha256": sha256(path / name)}
                     for name in checkpoint_files(args.world_size)}
            write_json(path / "deep_kv.json", {"format": "deep-kv-hf-checkpoint-v1",
                       "identity": self.identity, "update": state.global_step,
                       "history": self.history, "files": files})
            write_json(Path(args.output_dir) / "metrics.json", self.history)
            certified = sorted(
                (p for p in Path(args.output_dir).glob("checkpoint-*")
                 if p.name.removeprefix("checkpoint-").isdigit() and (p / "deep_kv.json").is_file()),
                key=lambda p: int(p.name.split("-")[-1]),
            )
            for old in certified[:-2]:
                shutil.rmtree(old)
        self.trainer.accelerator.wait_for_everyone()


def train(model, train_data, eval_data, recipe, output, identity, *, microbatch=1,
          mixed_precision=True, resume=False, stop_after=None):
    os.environ["WANDB_MODE"] = "offline"
    os.environ.setdefault("WANDB_PROJECT", "deep2shallow")
    recipe.validate()
    if identity.get("recipe") != asdict(recipe) or identity.get("arm") != model.arm:
        raise ValueError("Run identity differs from the requested arm/recipe")
    end = recipe.end_update(stop_after)
    output = Path(output)
    world, rank = topology()
    if (identity.get("world_size", world) != world or
            identity.get("mixed_precision", mixed_precision) != mixed_precision or
            identity.get("config", {}).get("microbatch", microbatch) != microbatch):
        raise ValueError("Runtime settings differ from run identity")
    cpu = next(model.parameters()).device.type == "cpu"
    args = training_arguments(recipe, output, microbatch, cpu=cpu, mixed_precision=mixed_precision)
    if args.world_size != world or (args.device.type == "cpu") != cpu:
        raise ValueError("Trainer device/topology differs from the requested runtime")
    checkpoint = None
    history = {"train": [], "evaluation": []}
    update = 0
    if resume:
        if (output / "complete.json").exists():
            raise ValueError("Resume requires an incomplete run")
        if json.loads((output / "run.json").read_text()) != identity:
            raise ValueError("Run receipt differs from checkpoint identity")
        candidates = [p for p in output.glob("checkpoint-*")
                      if p.name.removeprefix("checkpoint-").isdigit() and (p / "deep_kv.json").is_file()]
        if not candidates:
            raise ValueError("Resume requires a certified HF checkpoint")
        checkpoint = max(candidates, key=lambda p: int(p.name.split("-")[-1]))
        receipt = verify_checkpoint(checkpoint, identity, verify_hashes=rank == 0)
        update, history = receipt["update"], receipt["history"]
        if end < update:
            raise ValueError("Cutoff precedes checkpoint")
    elif rank == 0 and output.exists():
        raise FileExistsError(output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=resume)
        if not resume:
            write_json(output / "run.json", identity)
        (output / "stopped.json").unlink(missing_ok=True)
    if distributed():
        dist.barrier()
    with offline_wandb_run(args):
        callback = ExperimentCallback(recipe, identity, end, eval_data, history)
        trainer = DeepKVTrainer(model=model, args=args, train_dataset=train_data,
                                eval_dataset=Subset(eval_data, range(recipe.monitor_rows)),
                                data_collator=default_data_collator, compute_metrics=compute_metrics,
                                callbacks=[callback])
        callback.trainer = trainer
        if update < end:
            trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
            if trainer.state.global_step != end:
                raise ValueError("Trainer stopped outside the requested iteration")
            checkpoint = output / f"checkpoint-{end}"
        else:
            # Do not invoke Trainer.train at an already reached cutoff: it can take
            # another step before its stopping callback fires. Only restore/evaluate.
            trainer._load_from_checkpoint(str(checkpoint))
            trainer.state = TrainerState.load_from_json(str(checkpoint / "trainer_state.json"))
            if (not history["evaluation"] or history["evaluation"][-1]["update"] != end
                    or history["evaluation"][-1]["rows"] != recipe.eval_rows):
                trainer.evaluate(eval_dataset=eval_data)
                if rank == 0:
                    receipt["history"] = history
                    write_json(checkpoint / "deep_kv.json", receipt)
        trainer.accelerator.wait_for_everyone()
        if rank == 0:
            receipt = verify_checkpoint(checkpoint, identity)
            history = receipt["history"]
            if history["evaluation"][-1]["rows"] != recipe.eval_rows or history["evaluation"][-1]["update"] != end:
                raise ValueError("Missing fixed final evaluation")
            write_json(output / "metrics.json", history)
            status = {"status": "complete" if end == recipe.updates else "stopped",
                      "arm": model.arm, "name": NAMES[model.arm], "update": end,
                      "input_tokens": end * recipe.tokens_per_update,
                      "checkpoint": checkpoint.name,
                      "final_evaluation": history["evaluation"][-1],
                      "utc": datetime.now(timezone.utc).isoformat()}
            write_json(output / ("complete.json" if end == recipe.updates else "stopped.json"), status)
        trainer.accelerator.wait_for_everyone()
        return history
