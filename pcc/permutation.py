"""Deterministic within-global-update correction-target derangement.

Buckets: floor(10 * token_index / valid_context_length), crossed with teacher
norm quantiles over eligible targets in the global update. Equal norms share
a bucket. Permutation operates on CPU, independently of microbatch boundaries.
No singleton bucket can be deranged: fail rather than leave a semantic target
unchanged or silently merge/relax the preregistered buckets.
"""
import numpy as np
import torch
from .protocol import DATA_SEED


def permute_targets(targets, context, update):
    if targets.device.type != "cpu" or context.input_ids.device.type != "cpu":
        raise ValueError("Permutation pool must be the CPU global-update batch")
    eligible = context.targets()
    indices = torch.nonzero(eligible, as_tuple=False)
    if len(indices) < 2:
        raise ValueError("Not enough eligible permutation targets")
    rows, positions = indices.unbind(-1)
    values = targets[:, :-1][eligible].detach()
    norms = values.float().norm(dim=-1).numpy()
    if not np.isfinite(norms).all():
        raise ValueError("Nonfinite teacher correction norms")
    position_bins = (10 * positions // context.valid.sum(-1)[rows]).numpy()
    edges = np.quantile(norms, np.arange(1, 10) / 10)
    norm_bins = np.searchsorted(edges, norms, side="right")
    buckets = position_bins * 10 + norm_bins
    rng = np.random.default_rng(np.random.SeedSequence([DATA_SEED, update]))
    mapping = np.arange(len(values))
    for bucket in np.unique(buckets):
        members = np.flatnonzero(buckets == bucket)
        if len(members) < 2:
            raise ValueError(f"Cannot derange singleton target bucket {int(bucket)} in update {update}")
        shuffled = rng.permutation(members)
        mapping[shuffled] = np.roll(shuffled, 1)
    if np.any(mapping == np.arange(len(mapping))) or not np.array_equal(buckets, buckets[mapping]):
        raise RuntimeError("Invalid target derangement")
    result = targets.detach().clone()
    result[:, :-1][eligible] = values[torch.from_numpy(mapping)]
    return result, {"mapping": mapping, "buckets": buckets, "indices": indices.numpy(),
                    "norm_decile_edges": edges.tolist(), "seed": DATA_SEED, "update": update}
