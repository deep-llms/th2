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


def install(model):
    model.exit_options = ExitTrainingConfig(**model.config.bdot_exit_training)
    model.exit_options.validate(model)
    # CE sum, KL sum, number of scored targets; only actual input tokens count
    # toward the main training statistics and data progress.
    for name in ('exit_training_totals', 'logged_exit_training_totals', 'exit_metric_totals'):
        model.register_buffer(name, torch.zeros(3, dtype=torch.float64),
                              persistent=name != 'exit_metric_totals')


def loss_sums(student, teacher, labels, chunk_size=256, include_kl=True):
    """Exact masked CE/KL sums, FP32 vocabulary math, bounded live intermediates.

    Avoid reshaping [:, :-1] into a huge copied vocabulary tensor. A terminal
    ignored target per row aligns flattened logits with shifted labels. Loss
    checkpointing recomputes softmax chunks, not transformer forwards.
    """
    if student.shape != teacher.shape or student.shape[:2] != labels.shape:
        raise ValueError('Exit logits/teacher/labels shapes differ')
    target = F.pad(labels[:, 1:], (0, 1), value=-100).reshape(-1)
    student = student.flatten(0, 1)
    teacher = teacher.detach().flatten(0, 1)

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


def forward(model, arguments):
    opt = model.exit_options
    if (arguments.get('bdot_stop_depths') is not None or arguments.get('past_key_values') is not None
            or arguments.get('use_cache') or arguments.get('logits_to_keep', 0) != 0):
        raise ValueError('Dual-path training requires full logits and no external exits/cache')
    labels = arguments['labels']
    full = model._forward_once(**arguments)
    full_depths = model.stop_controller.last_depths
    full_routes = model.last_routes
    # No second native CE or z loss. This traversal supplies logits for the
    # independently masked CE/KL below, normalized by HF's SAME global count.
    auxiliary = dict(arguments, labels=None, _bdot_auxiliary=True,
                     bdot_stop_depths=torch.full_like(labels, opt.depth))
    early = model._forward_once(**auxiliary)
    sums = loss_sums(early.logits, full.logits, labels, opt.chunk_size, opt.kl_weight > 0)
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
