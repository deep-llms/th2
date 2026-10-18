"""Document boundaries for native HF Qwen3 + Trainer, using SDPA or FA4.

The model/optimizer/checkpoint implementation stays in Transformers. Only input
mask preparation and validation are specialized; HF handles global token counts.
"""
from dataclasses import dataclass

import torch
from transformers import Trainer, default_data_collator


@dataclass
class PackedCollator:
    isolate_documents: bool = True
    reset_document_positions: bool = True
    attention_backend: str = "sdpa"

    def __post_init__(self):
        if self.attention_backend not in ("sdpa", "fa4"):
            raise ValueError("attention_backend must be sdpa or fa4")
        if self.attention_backend == "fa4" and not self.isolate_documents:
            raise ValueError("FA4 currently requires document isolation")
        if self.reset_document_positions and not self.isolate_documents:
            raise ValueError("Position reset requires document isolation")

    def __call__(self, rows):
        batch = default_data_collator(rows)
        segments = batch.pop("segments")  # Reject old caches without real document IDs.
        if segments.shape != batch["input_ids"].shape:
            raise ValueError("Document IDs must match packed input shape")
        if not bool(batch["attention_mask"].eq(1).all()):
            raise ValueError("This collator expects full fixed-length blocks, without padding")
        starts = torch.ones_like(segments, dtype=torch.bool)
        starts[:, 1:] = segments[:, 1:] != segments[:, :-1]
        # Original source-row IDs are monotonic; never reconnect disjoint runs.
        if bool((segments < 0).any()) or bool((segments[:, 1:] < segments[:, :-1]).any()):
            raise ValueError("Document IDs must be nondecreasing within each block")
        offsets = torch.arange(segments.shape[1]).expand_as(segments)
        batch["position_ids"] = (offsets - offsets.masked_fill(~starts, 0).cummax(-1).values
                                 if self.reset_document_positions else offsets)
        # Unshifted labels: EOS stays in its document; the next document's first
        # token is never a target of the preceding EOS. Label zero is unscored.
        batch["labels"] = batch["labels"].clone()
        batch["labels"][:, 0] = -100
        if self.isolate_documents:
            batch["labels"].masked_fill_(starts, -100)
            if self.attention_backend == "fa4":
                from packed_fa4 import document_layout
                batch["fa4_cu_seqlens"], batch["fa4_max_seqlen"] = document_layout(segments)
            else:
                batch["segments"] = segments
        return batch


def model_inputs(inputs):
    """Called after device transfer; avoid CPU/prefetch copies of B*L*L masks."""
    inputs = dict(inputs)
    if "fa4_cu_seqlens" in inputs:
        if "segments" in inputs:
            raise ValueError("Cannot combine FA4 offsets and dense document masks")
        # Native Qwen3 accepts prepared per-layer-type masks. FA4 uses offsets
        # instead; bypass HF's dense/position-inferred mask construction.
        inputs["attention_mask"] = {"full_attention": None}
        inputs["use_cache"] = False
        return inputs
    segments = inputs.pop("segments", None)
    if segments is not None:
        length = segments.shape[1]
        allowed = segments[:, :, None].eq(segments[:, None, :])
        allowed &= torch.ones(length, length, dtype=torch.bool, device=segments.device).tril()
        inputs["attention_mask"] = allowed[:, None]  # SDPA True means visible.
    return inputs


class PackedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.data_collator, "isolate_documents", False):
            backend = self.data_collator.attention_backend
            expected = "template_fa4" if backend == "fa4" else "sdpa"
            if self.model.config._attn_implementation != expected:
                raise ValueError("Model attention backend must match PackedCollator")
            if backend == "fa4" and (self.accelerator.split_batches or self.accelerator.dispatch_batches):
                raise ValueError("FA4 offsets require per-rank collation: split_batches and dispatch_batches must be false")
        if self.args.remove_unused_columns:
            raise ValueError("PackedTrainer requires remove_unused_columns=False to retain document IDs")
        if not self.model_accepts_loss_kwargs or not self.args.average_tokens_across_devices:
            raise ValueError("Packed training requires HF token-normalized loss across accumulation and ranks")
        if self.label_smoother is not None or self.compute_loss_func is not None:
            raise ValueError("Custom loss normalization needs separate validation; use native causal LM loss")

    def _prepare_inputs(self, inputs):
        return model_inputs(super()._prepare_inputs(inputs))

    def _get_num_items_in_batch(self, batch_samples, device):
        count = super()._get_num_items_in_batch(batch_samples, device)
        # A microbatch/rank may have zero targets while other microbatches/ranks
        # have some. Only reject an entirely unscorable global update.
        if count is not None and bool((count == 0).any()):
            raise ValueError("Global accumulation window has no within-document prediction targets")
        return count

    def _validate_packed_evaluation(self):
        # Independent replicas allow uneven shards without DDP forward collectives.
        # Sharded models and nonstandard objectives need a different evaluator.
        if (self.is_deepspeed_enabled or self.is_fsdp_enabled or self.args.n_gpu > 1
                or str(self.accelerator.distributed_type).split(".")[-1]
                not in {"NO", "MULTI_CPU", "MULTI_GPU"}):
            raise ValueError("Packed evaluation supports single-device and ordinary CPU/CUDA DDP only")
        if self.args.torch_compile or self.args.bf16_full_eval or self.args.fp16_full_eval:
            raise ValueError("Packed evaluation uses training precision; full-eval casting/compile are not validated")
        if (self.args.load_best_model_at_end or self.args.lr_scheduler_type.value
                in {"reduce_lr_on_plateau", "greedy"}):
            raise ValueError("Monitoring-only validation cannot select weights or change the LR schedule")
        if self.compute_metrics is not None or self.preprocess_logits_for_metrics is not None:
            raise ValueError("Packed evaluation returns loss/PPL only; custom prediction metrics need separate validation")
        if "schedule_free" in str(self.args.optim).lower():
            raise ValueError("Schedule-free optimizer evaluation requires separate validation")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        from packed_evaluation import preserve_rng_and_mode
        self._validate_packed_evaluation()
        # Keep native HF timing, logging, metric callbacks and checkpoint flow.
        with preserve_rng_and_mode(self.model_wrapped):
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def get_eval_dataloader(self, eval_dataset=None):
        from packed_evaluation import packed_eval_loader
        dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if isinstance(dataset, str) and isinstance(self.eval_dataset, dict):
            dataset = self.eval_dataset[dataset]
        if dataset is None:
            raise ValueError("Packed evaluation requires a held-out dataset")
        return packed_eval_loader(dataset, self.data_collator, self.accelerator, self.args.eval_batch_size)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None,
                        ignore_keys=None, metric_key_prefix="eval"):
        from packed_evaluation import score_packed_loader
        from transformers.trainer_utils import EvalLoopOutput
        self._validate_packed_evaluation()
        # Retain Accelerate's precision wrapper; removing it mutates model.forward.
        model = self.accelerator.unwrap_model(self.model_wrapped)
        self.callback_handler.eval_dataloader = dataloader
        metrics = score_packed_loader(model, dataloader, self.accelerator)
        return EvalLoopOutput(predictions=None, label_ids=None,
            metrics={f"{metric_key_prefix}_{key}": value for key, value in metrics.items()},
            num_samples=metrics["num_sequences"])
