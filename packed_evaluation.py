"""Shared packed validation scoring; no optimizer, scheduler or DDP forwards."""
from contextlib import contextmanager
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from packed_training import model_inputs


@contextmanager
def preserve_rng_and_mode(model=None):
    """Validation (including dataloader/callback work) must not consume train RNG."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for module in model.modules()] if model is not None else []
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def packed_eval_loader(dataset, collator, accelerator, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    # Private loader RNG and exact, unpadded shards; empty ranks are allowed.
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collator,
        sampler=range(accelerator.process_index, len(dataset), accelerator.num_processes),
        drop_last=False, num_workers=0, generator=torch.Generator().manual_seed(0))


def summed_nll(logits, labels, chunk_size=1024):
    """FP32 CE, FP64 summation; never average batch losses or retain all logits.

    Chunk the vocabulary-wide FP32 conversion, avoiding an extra full-size copy
    of BF16/FP16 logits. The model still emits one microbatch of logits.
    """
    total = torch.zeros((), dtype=torch.float64, device=logits.device)
    for row in range(labels.shape[0]):
        for start in range(0, labels.shape[1] - 1, chunk_size):
            end = min(start + chunk_size, labels.shape[1] - 1)
            losses = F.cross_entropy(logits[row, start:end].float(), labels[row, start + 1:end + 1],
                                     ignore_index=-100, reduction="none")
            total += losses.double().sum()
    return total


@torch.inference_mode()
def evaluate_packed(model, dataset, collator, accelerator, batch_size=1, *, batch_observer=None):
    loader = packed_eval_loader(dataset, collator, accelerator, batch_size)
    return score_packed_loader(model, loader, accelerator, batch_observer=batch_observer)


@torch.inference_mode()
def score_packed_loader(model, loader, accelerator, *, batch_observer=None):
    backend = loader.collate_fn.attention_backend
    expected = "template_" + backend if backend in ("fa4", "fa2") else "sdpa"
    if model.config._attn_implementation != expected:
        raise ValueError("Model and evaluation collator attention backends must match")
    model.eval()
    if hasattr(model, "eval_counter"):
        model.eval_counter = 0  # Repeatable evaluation-only routing stream.
    loss_sum = torch.zeros((), dtype=torch.float64, device=accelerator.device)
    num_tokens = num_sequences = 0
    route_totals = torch.zeros(4, device=accelerator.device, dtype=torch.int64)
    for batch in loader:
        count = int(batch["labels"][:, 1:].ne(-100).sum())  # CPU, before transfer.
        num_sequences += batch["input_ids"].shape[0]
        num_tokens += count
        if not count and batch_observer is None and not hasattr(model, 'eval_counter'):
            # B-DOT always visits all-boundary batches so standalone and route
            # study evaluations have identical RNG advancement and route scope.
            continue
        inputs = model_inputs({k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
                               for k, v in batch.items()})
        labels = inputs.pop("labels")
        with accelerator.autocast():
            logits = model(**inputs).logits
        if batch_observer is not None:
            batch_observer(batch, model)
        if getattr(model, "capture_routes", False) and model.last_routes is not None:
            routes = model.last_routes
            steps = routes.shape[0]
            off = routes.ne(torch.arange(steps, device=routes.device)[:, None, None]).sum()
            visits = torch.nn.functional.one_hot(routes, num_classes=steps).sum(0)
            # Phase A's default schedule visits every module once. The spec's
            # reuse rate counts tokens with any repeated module, not repeated
            # decisions divided by all decisions.
            reused_tokens = visits.gt(1).any(-1).sum()
            route_totals += torch.stack((off, reused_tokens, off.new_tensor(routes.numel()),
                                         off.new_tensor(routes.shape[1] * routes.shape[2])))
        loss_sum += summed_nll(logits, labels)
        del logits
    loss_sum = float(accelerator.reduce(loss_sum, reduction="sum"))
    counts = accelerator.reduce(torch.tensor([num_tokens, num_sequences], device=accelerator.device,
                                            dtype=torch.int64), reduction="sum").tolist()
    if counts[1] != len(loader.dataset):
        raise RuntimeError("Distributed evaluation did not score each packed sequence exactly once")
    if counts[0] == 0 or not math.isfinite(loss_sum):
        raise ValueError("Evaluation has no scored targets or nonfinite loss")
    loss = loss_sum / counts[0]
    try:
        ppl = math.exp(loss)
    except OverflowError as error:
        raise ValueError(f"Perplexity overflow at loss={loss}") from error
    result = dict(loss=loss, perplexity=ppl, nll_sum=loss_sum,
                  num_tokens=counts[0], num_sequences=counts[1])
    if getattr(model, "capture_routes", False):
        off, reused_tokens, decisions, input_tokens = accelerator.reduce(route_totals, reduction="sum").tolist()
        result.update(off_default_share=off/max(decisions, 1), reuse_rate=reused_tokens/max(input_tokens, 1),
                      routing_decisions=decisions, routing_input_tokens=input_tokens)
    return result
