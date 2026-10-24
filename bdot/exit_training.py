"""Full-path supervision plus an actual held-state early path, within one DDP forward.

The original Trainer still owns accumulation, global target counts, AMP and
resume. The auxiliary traversal uses the same routing seed without advancing
the checkpointed counter or double-counting data/load statistics. Balancing and
router z regularization use the full path only. No teacher gradients from KL.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ExitTrainingConfig:
    depth: int = 6
    ce_weight: float = .5
    kl_weight: float = 0.
    chunk_size: int = 256
    # Explicit opt-ins until the target GPU/backend has passed equivalence and
    # capacity checks. Old checkpoints retain their original execution path.
    ce_backend: str = 'chunked'
    shared_prefix: bool = False

    def validate(self, model):
        ctl = getattr(model, 'stop_controller', None)
        if (ctl is None or ctl.options.train_random_prob != 0
                or tuple(ctl.options.exit_steps) != (self.depth,)
                or type(self.depth) is not int or not 1 <= self.depth < model.options.S):
            raise ValueError('Dual-path training needs one eligible exit and zero random STOP')
        if (not all(math.isfinite(w) and w >= 0 for w in (self.ce_weight, self.kl_weight))
                or self.ce_weight + self.kl_weight <= 0
                or type(self.chunk_size) is not int or self.chunk_size < 1):
            raise ValueError('Invalid exit loss weights/chunk size')
        if (self.ce_backend not in ('chunked', 'native') or type(self.shared_prefix) is not bool
                or (self.ce_backend == 'native' and self.kl_weight != 0)):
            raise ValueError('Native exit CE requires zero KL weight; invalid execution options')


def install(model):
    model.exit_options = ExitTrainingConfig(**model.config.bdot_exit_training)
    model.exit_options.validate(model)
    # CE sum, KL sum, number of scored targets; only actual input tokens count
    # toward the main training statistics and data progress.
    for name in ('exit_training_totals', 'logged_exit_training_totals', 'exit_metric_totals'):
        model.register_buffer(name, torch.zeros(3, dtype=torch.float64),
                              persistent=name != 'exit_metric_totals')


def loss_sums(student, teacher, labels, chunk_size=256, include_kl=True, *, ce_backend='chunked'):
    """Exact masked CE/KL sums with FP32 vocabulary math.

    Avoid reshaping [:, :-1] into a huge copied vocabulary tensor. A terminal
    ignored target per row aligns flattened logits with shifted labels. Loss
    checkpointing recomputes softmax chunks, not transformer forwards. The
    native CE-only option trades additional live softmax memory for throughput.
    """
    if student.shape != teacher.shape or student.shape[:2] != labels.shape:
        raise ValueError('Exit logits/teacher/labels shapes differ')
    target = F.pad(labels[:, 1:], (0, 1), value=-100).reshape(-1)
    student = student.flatten(0, 1)
    teacher = teacher.detach().flatten(0, 1)
    if ce_backend == 'native':
        if include_kl:
            raise ValueError('Native exit CE cannot omit a requested KL objective')
        # Keep the terminal ignored target, avoiding a copied [:, :-1] logits
        # tensor. One native CE replaces many slice-backward allocations and
        # checkpoint replays; it uses more live softmax memory.
        with torch.autocast(device_type=student.device.type, enabled=False):
            ce = F.cross_entropy(student.float(), target, ignore_index=-100, reduction='sum')
        return torch.stack((ce, ce.new_zeros(())))
    if ce_backend != 'chunked':
        raise ValueError('Unknown exit CE backend')

    def chunk(x, t, y):
        with torch.autocast(device_type=x.device.type, enabled=False):
            logp = F.log_softmax(x.float(), dim=-1)
            ce = F.nll_loss(logp, y, ignore_index=-100, reduction='sum')
            kl = ce.new_zeros(())
            if include_kl:
                logq = F.log_softmax(t.float(), dim=-1)
                per_token = (logq.exp() * (logq - logp)).sum(-1)
                kl = per_token.masked_fill(y == -100, 0).sum()
            return torch.stack((ce, kl))

    sums = []
    for start in range(0, len(target), chunk_size):
        args = (student[start:start+chunk_size], teacher[start:start+chunk_size],
                target[start:start+chunk_size])
        sums.append(checkpoint(chunk, *args, use_reentrant=False, preserve_rng_state=False)
                    if torch.is_grad_enabled() and student.requires_grad else chunk(*args))
    return torch.stack(sums).sum(0)


def shared_prefix_forward(model, arguments):
    """Run the identical prefix once; branch through the ordinary coda twice.

    All auxiliary tokens exit at the same depth, with no training cache. Later
    middle-step K/V writes have no consumers and cannot affect the coda. Mixed
    exits and cached inference still use the original held-state implementation.
    The captured tensor stays attached so BOTH losses train the common prefix.
    """
    captured = []

    def capture(_module, _args, kwargs, hidden):
        captured.append((hidden, kwargs.copy()))

    handle = model.routed_steps[model.exit_options.depth - 1].register_forward_hook(
        capture, with_kwargs=True)
    try:
        full = model._forward_once(**arguments)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError('Expected one full-path prefix capture')
    hidden, kwargs = captured.pop()
    if kwargs.get('past_key_values') is not None or kwargs.get('use_cache'):
        raise ValueError('Sharing the exit prefix requires cache-free training')
    # Reuse the actual native mask/RoPE and packed offsets. The coda is ordinary
    # Qwen3 and has no routing decisions or mutable RouteContext to update.
    kwargs.pop('bdot_context', None)
    for layer in model.model.layers[model.options.n_stem + model.options.S:]:
        hidden = layer(hidden, **kwargs)
    return full, model.lm_head(model.model.norm(hidden))


def forward(model, arguments):
    opt = model.exit_options
    if (arguments.get('bdot_stop_depths') is not None or arguments.get('past_key_values') is not None
            or arguments.get('use_cache') or arguments.get('logits_to_keep', 0) != 0):
        raise ValueError('Dual-path training requires full logits and no external exits/cache')
    labels = arguments['labels']
    if opt.shared_prefix:
        full, early_logits = shared_prefix_forward(model, arguments)
    else:
        full = model._forward_once(**arguments)
    full_depths = model.stop_controller.last_depths
    full_routes = model.last_routes
    # No second native CE or z loss. This traversal supplies logits for the
    # independently masked CE/KL below, normalized by HF's SAME global count.
    if not opt.shared_prefix:
        auxiliary = dict(arguments, labels=None, _bdot_auxiliary=True,
                         bdot_stop_depths=torch.full_like(labels, opt.depth))
        early_logits = model._forward_once(**auxiliary).logits
    sums = loss_sums(early_logits, full.logits, labels, opt.chunk_size, opt.kl_weight > 0,
                     ce_backend=opt.ce_backend)
    count = labels[:, 1:].ne(-100).sum()
    denominator = arguments.get('num_items_in_batch')
    if denominator is None:
        denominator = count.clamp_min(1)
    full.loss = full.loss + (opt.ce_weight*sums[0] + opt.kl_weight*sums[1]) / denominator
    with torch.no_grad():
        model.exit_metric_totals.add_(torch.stack((sums[0].detach(), sums[1].detach(), count)).double())
    # Public output, captured route/depth diagnostics, and main LM metrics all
    # describe the full path. The auxiliary path must not silently replace them.
    model.stop_controller.last_depths = full_depths
    model.last_routes = full_routes
    return full


@torch.no_grad()
def finish_update(model):
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(model.exit_metric_totals)
    model.exit_training_totals.add_(model.exit_metric_totals)
    model.exit_metric_totals.zero_()
