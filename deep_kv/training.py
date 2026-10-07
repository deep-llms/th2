"""Only the HF Trainer adaptations needed by the Deep-KV variants."""
from contextlib import contextmanager
from functools import partial
import os
from pathlib import Path

import numpy as np
import torch
from transformers import Trainer, TrainerCallback
from .model import Context
from .sdpa_audit import trainer_audit
from . import BOTTLENECK_ARMS, code_loss_weight


def summarize_statistics(statistics, kv_loss_weight=1.0, arm=None, *, evaluation=False):
    values = np.asarray(statistics, dtype=np.float64)
    if arm in BOTTLENECK_ARMS:
        lm, targets, extract, queries, align, no_message, tokens = values.sum(axis=0).tolist()
        if targets <= 0 or tokens <= 0 or queries < 0 or not np.isfinite(values).all():
            raise ValueError("Nonfinite loss or empty token statistics")
        extract, align = extract / max(queries, 1), align / tokens
        consumer = arm.startswith("Consumer")
        result = {"lm_loss": lm / targets, "loss_use" if consumer else "loss_extract": extract,
                  "loss_align": align, "objective": lm / targets + extract + code_loss_weight(arm) * align,
                  "extractor_targets": int(queries), "input_tokens": int(tokens),
                  "target_tokens": int(targets), "rows": len(values)}
        if consumer and evaluation:
            result.update(loss_use_no_message=no_message / max(queries, 1),
                          message_ce_gain=no_message / max(queries, 1) - extract)
        return result
    if arm in ("F", "G"):
        lm, targets, route, msg, queries, tokens = values.sum(axis=0).tolist()
        if targets <= 0 or tokens <= 0 or queries < 0 or not np.isfinite(values).all():
            raise ValueError("Nonfinite loss or empty token statistics")
        route, msg = route / max(queries, 1), msg / max(queries, 1)
        return {"lm_loss": lm / targets, "loss_route": route, "loss_msg": msg,
                "objective": lm / targets + kv_loss_weight * (route + (msg if arm == "G" else 0)),
                "route_queries": int(queries), "input_tokens": int(tokens),
                "target_tokens": int(targets), "rows": len(values)}
    lm, targets, k, v, tokens = values.sum(axis=0).tolist()
    if targets <= 0 or tokens <= 0 or not np.isfinite(values).all():
        raise ValueError("Nonfinite loss or empty token statistics")
    return {"lm_loss": lm / targets, "loss_k": k / tokens, "loss_v": v / tokens,
            "objective": lm / targets + kv_loss_weight * (k + v) / (2 * tokens),
            "input_tokens": int(tokens), "target_tokens": int(targets), "rows": len(values)}


def compute_metrics(prediction, kv_loss_weight=1.0, arm=None):
    result = summarize_statistics(prediction.predictions, kv_loss_weight, arm, evaluation=True)
    result["loss"] = result["lm_loss"] if arm in BOTTLENECK_ARMS else result["objective"]
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



def save_module(trainer, output_dir=None, state_dict=None):
    """HF nn.Module saving with explicit copies of genuinely shared storage."""
    state = dict(trainer.model.state_dict() if state_dict is None else state_dict)
    seen = set()
    for name, value in state.items():
        key = (value.device, value.untyped_storage().data_ptr())
        if key in seen:
            state[name] = value.clone()
        seen.add(key)
    Trainer._save(trainer, output_dir, state_dict=state)


class DeepKVTrainer(Trainer):
    """Keep the HF loop; customize loss and compact evaluation."""

    def __init__(self, *args, **kwargs):
        import accelerate
        import transformers
        if transformers.__version__ != "5.9.0" or accelerate.__version__ != "1.13.0":
            raise ValueError("Use transformers==5.9.0 and accelerate==1.13.0 for this Trainer integration")
        super().__init__(*args, **kwargs)
        # Bind the project's evaluation metric to the same coefficient as
        # training. Keep explicitly supplied unrelated metric functions intact.
        if self.compute_metrics is compute_metrics:
            self.compute_metrics = partial(compute_metrics, kv_loss_weight=self.model.kv_loss_weight,
                                           arm=self.model.arm)
        # Legacy arms return microbatch means. Bottleneck arms consume the three
        # global denominators below, so Trainer must not divide by GAS again.
        self.model_accepts_loss_kwargs = self.model.bottleneck
        self.step_totals = None
        self._sdpa_seen = set()
        self._sdpa_receipts = []
        self._active_sdpa_audit = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        with trainer_audit(self, "train"):
            return super().training_step(model, inputs, num_items_in_batch)

    def summarize_statistics(self, values):
        return summarize_statistics(values, self.model.kv_loss_weight, self.model.arm)

    @staticmethod
    def context(inputs):
        ids = inputs["input_ids"]
        return Context(ids, inputs.get("attention_mask", torch.ones_like(ids)).bool(),
                       inputs.get("position_ids", torch.arange(ids.shape[1], device=ids.device).expand_as(ids)),
                       inputs.get("segments"), inputs.get("labels"))

    def _get_num_items_in_batch(self, batch_samples, device):
        if not self.model.bottleneck:
            return super()._get_num_items_in_batch(batch_samples, device)
        if not batch_samples:
            return None
        # The three losses have DIFFERENT denominators. Count across the entire
        # accumulation window and all DDP ranks, including uneven valid masks.
        counts = torch.zeros(3, dtype=torch.long, device=device)
        for inputs in batch_samples:
            context = self.context(inputs)
            eligible = context.consumer_targets() if self.model.consumer_aware else context.targets()
            counts += torch.stack((context.targets().sum(), eligible.sum(), context.valid.sum())).to(device)
        return self.accelerator.reduce(counts, reduction="sum")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        context = self.context(inputs)
        outputs = model(context)
        if self.model.bottleneck:
            counts = (num_items_in_batch if num_items_in_batch is not None else
                      torch.stack((outputs["lm_count"], outputs["extractor_count"], outputs["align_count"])))
            lm_n, extract_n, align_n = counts.clamp_min(1)
            loss = outputs["lm_sum"] / lm_n + outputs["extractor_sum"] / extract_n
            if self.model.code_loss_weight:
                loss = loss + self.model.code_loss_weight * outputs["align_sum"] / align_n
            if num_items_in_batch is not None:
                # Trainer skips its accumulation divisor; compensate only for
                # DDP's gradient average. Our denominators already cover GAS.
                loss = loss * self.accelerator.num_processes
        elif self.model.functional_loss:
            auxiliary = outputs["route_sum"]
            if self.model.arm == "G":
                auxiliary = auxiliary + outputs["msg_sum"]
            loss = outputs["lm_sum"] / outputs["lm_count"] + self.model.kv_loss_weight * auxiliary / outputs["route_count"].clamp_min(1)
        else:
            loss = (outputs["lm_sum"] / outputs["lm_count"] +
                    self.model.kv_loss_weight * (outputs["k_sum"] + outputs["v_sum"]) / (2 * outputs["kv_count"]))
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite objective")
        if model.training and self.is_in_train:
            totals = outputs["statistics"].sum(dim=0)
            self.step_totals = totals if self.step_totals is None else self.step_totals + totals
        if self._active_sdpa_audit is not None and loss.requires_grad:
            loss.register_hook(self._active_sdpa_audit.backward_marker)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with trainer_audit(self, "eval"), torch.no_grad(), self.compute_loss_context_manager():
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
        save_module(self, output_dir, state_dict)

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



class PilotCallback(TrainerCallback):
    """Log component losses and stop at the same update without changing LR."""
    def __init__(self, end, tokens_per_update):
        self.end, self.tokens_per_update = end, tokens_per_update
        self.trainer = None
        self.latest = {}

    def on_step_end(self, args, state, control, **kwargs):
        totals = self.trainer.step_totals
        self.trainer.step_totals = None
        if totals is None:
            raise ValueError("Missing training statistics")
        totals = self.trainer.accelerator.reduce(totals, reduction="sum")
        if int(totals[-1]) != self.tokens_per_update:
            raise ValueError("Incomplete global token batch; check dataset capacity")
        self.latest = self.trainer.summarize_statistics(totals.cpu().numpy()[None])
        if state.global_step >= self.end:
            control.should_training_stop = control.should_save = True
        return control

    def on_log(self, args, state, control, logs, **kwargs):
        if "loss" in logs:
            logs.update({f"step_{k}": self.latest[k] for k in
                         ("lm_loss", "loss_k", "loss_v", "loss_route", "loss_msg",
                          "loss_extract", "loss_use", "loss_align") if k in self.latest})
