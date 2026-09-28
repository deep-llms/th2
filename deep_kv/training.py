"""Only the HF Trainer adaptations needed by the four-arm model."""
from contextlib import contextmanager
from functools import partial
import os
from pathlib import Path

import numpy as np
import torch
from transformers import Trainer, TrainerCallback
from .model import Context


def summarize_statistics(statistics, kv_loss_weight=1.0):
    values = np.asarray(statistics, dtype=np.float64)
    lm, targets, k, v, tokens = values.sum(axis=0).tolist()
    if targets <= 0 or tokens <= 0 or not np.isfinite(values).all():
        raise ValueError("Nonfinite loss or empty token statistics")
    return {"lm_loss": lm / targets, "loss_k": k / tokens, "loss_v": v / tokens,
            "objective": lm / targets + kv_loss_weight * (k + v) / (2 * tokens),
            "input_tokens": int(tokens), "target_tokens": int(targets), "rows": len(values)}


def compute_metrics(prediction, kv_loss_weight=1.0):
    result = summarize_statistics(prediction.predictions, kv_loss_weight)
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
            self.compute_metrics = partial(compute_metrics, kv_loss_weight=self.model.kv_loss_weight)
        # We return a microbatch mean, not a sum normalized by HF's label count.
        # Trainer divides once by accumulation; DDP averages once across ranks.
        self.model_accepts_loss_kwargs = False
        self.step_totals = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ids = inputs["input_ids"]
        context = Context(ids, torch.ones_like(ids, dtype=torch.bool),
                          torch.arange(ids.shape[1], device=ids.device).expand_as(ids))
        outputs = model(context)
        loss = (outputs["lm_sum"] / outputs["lm_count"] +
                self.model.kv_loss_weight * (outputs["k_sum"] + outputs["v_sum"]) / (2 * outputs["kv_count"]))
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
        self.latest = summarize_statistics(totals.cpu().numpy()[None], self.trainer.model.kv_loss_weight)
        if state.global_step >= self.end:
            control.should_training_stop = control.should_save = True
        return control

    def on_log(self, args, state, control, logs, **kwargs):
        if "loss" in logs:
            logs.update({f"step_{k}": self.latest[k] for k in ("lm_loss", "loss_k", "loss_v") if k in self.latest})
