"""Offline initialization utilities for post-hoc compression baselines.

These routines are intentionally separate from forward modules.  P-VQ and
GroupReduce are multi-stage compression procedures in their original papers;
constructing random compact factors inside a training loop is not a faithful
reproduction of either method.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from .compressed_baselines import GroupReduceEmbed, PVQEmbed


def tensor_sha256(tensor):
    """Hash tensor values, shape, and dtype without changing the tensor."""
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_frequency_counts(path, vocab_size, key="counts", pseudocount=1.0):
    """Load and validate the frequency vector used by GroupReduce."""
    if pseudocount < 0:
        raise ValueError("pseudocount must be nonnegative")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Frequency file not found: {source}")
    if source.suffix == ".npz":
        with np.load(source) as archive:
            if key not in archive:
                raise KeyError(f"{source} has no array named {key!r}")
            values = archive[key]
        counts = torch.as_tensor(values, dtype=torch.float64)
    elif source.suffix in {".pt", ".pth"}:
        value = torch.load(source, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            if key not in value:
                raise KeyError(f"{source} has no tensor named {key!r}")
            value = value[key]
        counts = torch.as_tensor(value, dtype=torch.float64)
    else:
        raise ValueError("frequency file must be .npz, .pt, or .pth")
    if counts.ndim != 1 or counts.numel() != vocab_size:
        raise ValueError(
            f"frequency counts must have shape ({vocab_size},), got "
            f"{tuple(counts.shape)}"
        )
    if not torch.isfinite(counts).all() or torch.any(counts < 0):
        raise ValueError("frequency counts must be finite and nonnegative")
    return counts.clamp_min(float(pseudocount))


def frequency_rank_order(counts):
    """Return deterministic high-to-low frequency order (token-id ties)."""
    counts = torch.as_tensor(counts, dtype=torch.float64).cpu()
    if counts.ndim != 1 or not torch.isfinite(counts).all():
        raise ValueError("counts must be a finite one-dimensional tensor")
    if torch.any(counts < 0):
        raise ValueError("counts must be nonnegative")
    return torch.argsort(counts, descending=True, stable=True)


def frequency_group_ids(counts, num_groups):
    """Stable frequency-rank bins used to initialize GroupReduce."""
    counts = torch.as_tensor(counts, dtype=torch.float64).cpu()
    vocab_size = counts.numel()
    if not 0 < num_groups <= vocab_size:
        raise ValueError("num_groups must be between 1 and vocabulary size")
    # Stable sorting makes token id the deterministic tie-breaker.
    order = frequency_rank_order(counts)
    base, remainder = divmod(vocab_size, num_groups)
    sizes = [base + (group < remainder) for group in range(num_groups)]
    group_ids = torch.empty(vocab_size, dtype=torch.long)
    start = 0
    for group, size in enumerate(sizes):
        group_ids[order[start:start + size]] = group
        start += size
    return group_ids


def frequency_group_ids_from_populations(counts, populations):
    """Assign frequency-ranked tokens to blocks of explicit populations."""
    counts = torch.as_tensor(counts, dtype=torch.float64).cpu()
    populations = tuple(int(size) for size in populations)
    if not populations or any(size <= 0 for size in populations):
        raise ValueError("populations must contain positive sizes")
    if sum(populations) != counts.numel():
        raise ValueError(
            f"population sum {sum(populations)} does not equal vocabulary "
            f"size {counts.numel()}"
        )
    order = frequency_rank_order(counts)
    group_ids = torch.empty(counts.numel(), dtype=torch.long)
    start = 0
    for group, size in enumerate(populations):
        group_ids[order[start:start + size]] = group
        start += size
    return group_ids


def file_sha256(path, chunk_size=1024 * 1024):
    """Hash an input artifact without reading the whole file at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def group_parameter_count(group_ids, ranks, embed_dim):
    group_ids = torch.as_tensor(group_ids, dtype=torch.long).cpu()
    sizes = torch.bincount(group_ids, minlength=len(ranks)).tolist()
    return sum(
        int(rank) * (int(size) + int(embed_dim))
        for size, rank in zip(sizes, ranks)
    )


def allocate_frequency_proportional_ranks(
    counts,
    group_ids,
    embed_dim,
    target_params,
):
    """Resolve the paper's proportional-rank rule under a fixed budget.

    The paper defines each rank as proportional to a group's average frequency
    but does not specify rounding under an exact parameter budget.  We choose
    the largest deterministic scale whose factor count does not exceed the
    requested budget and report the resolved ranks explicitly.
    """
    counts = torch.as_tensor(counts, dtype=torch.float64)
    group_ids = torch.as_tensor(
        group_ids, dtype=torch.long, device=counts.device
    )
    if counts.ndim != 1 or group_ids.shape != counts.shape or counts.numel() == 0:
        raise ValueError("counts and group_ids must be non-empty matching vectors")
    if not torch.isfinite(counts).all() or torch.any(counts < 0):
        raise ValueError("counts must be finite and nonnegative")
    if group_ids.min().item() < 0:
        raise ValueError("group_ids must be nonnegative")
    num_groups = int(group_ids.max().item()) + 1
    sizes = torch.bincount(group_ids, minlength=num_groups)
    if torch.any(sizes == 0):
        raise ValueError("group_ids must use every group id contiguously")
    averages = torch.stack([
        counts[group_ids == group].mean() for group in range(num_groups)
    ])
    if averages.min().item() <= 0:
        raise ValueError(
            "frequency-proportional rank allocation requires every group to "
            "have positive average frequency; use a positive pseudocount or "
            "provide explicit --groupreduce_ranks"
        )
    ratios = averages / averages.min()
    caps = torch.minimum(
        sizes, torch.full_like(sizes, int(embed_dim))
    ).to(torch.long)

    minimum = sum((sizes + embed_dim).tolist())
    maximum = sum(((sizes + embed_dim) * caps).tolist())
    if target_params < minimum:
        raise ValueError(
            f"target_params={target_params} is below rank-1 minimum {minimum}"
        )
    if target_params >= maximum:
        return tuple(int(value) for value in caps.tolist())

    def ranks_at(scale):
        ranks = torch.round(ratios * scale).to(torch.long)
        return torch.minimum(caps, ranks.clamp_min(1))

    def params_at(scale):
        ranks = ranks_at(scale)
        return int(((sizes + embed_dim) * ranks).sum().item()), ranks

    low, high = 0.0, 1.0
    while params_at(high)[0] <= target_params:
        high *= 2.0
    best_ranks = ranks_at(low)
    for _ in range(100):
        middle = (low + high) / 2.0
        params, ranks = params_at(middle)
        if params <= target_params:
            low = middle
            best_ranks = ranks
        else:
            high = middle
    return tuple(int(value) for value in best_ranks.tolist())


@torch.no_grad()
def weighted_low_rank_factors(matrix, weights, rank):
    """Closed-form row-weighted SVD factors from GroupReduce Eq. (2-3)."""
    matrix = torch.as_tensor(matrix, dtype=torch.float32)
    weights = torch.as_tensor(
        weights, dtype=torch.float32, device=matrix.device
    )
    if matrix.ndim != 2 or weights.ndim != 1 or weights.numel() != matrix.size(0):
        raise ValueError("matrix/weights have incompatible shapes")
    if torch.any(weights <= 0) or not torch.isfinite(weights).all():
        raise ValueError("weighted SVD requires finite positive row weights")
    max_rank = min(matrix.shape)
    if not 0 < rank <= max_rank:
        raise ValueError(f"rank must be in [1, {max_rank}], got {rank}")

    sqrt_weights = weights.sqrt()
    weighted = matrix * sqrt_weights.unsqueeze(1)
    left_singular, singular_values, right_transpose = torch.linalg.svd(
        weighted, full_matrices=False
    )
    left = (
        left_singular[:, :rank] * singular_values[:rank]
    ) / sqrt_weights.unsqueeze(1)
    right = right_transpose[:rank].T.contiguous()
    return left, right


@torch.no_grad()
def initialize_groupreduce_from_dense(
    dense_weight,
    counts,
    group_ids,
    ranks,
):
    """Apply weighted block-SVD and return a strict GroupReduce state dict."""
    dense_weight = torch.as_tensor(dense_weight, dtype=torch.float32)
    counts = torch.as_tensor(counts, dtype=torch.float32)
    group_ids = torch.as_tensor(group_ids, dtype=torch.long)
    if dense_weight.ndim != 2:
        raise ValueError("dense_weight must be a matrix")
    vocab_size, embed_dim = dense_weight.shape
    if counts.shape != (vocab_size,) or group_ids.shape != (vocab_size,):
        raise ValueError("counts/group_ids must have one entry per token")

    module = GroupReduceEmbed(
        vocab_size, embed_dim, group_ranks=ranks, group_ids=group_ids
    )
    for group, rank in enumerate(ranks):
        token_ids = module.token_ids_for_group(group)
        device_token_ids = token_ids.to(dense_weight.device)
        left, right = weighted_low_rank_factors(
            dense_weight[device_token_ids], counts[device_token_ids], int(rank)
        )
        module.left_factors[group].copy_(left.cpu())
        module.right_factors[group].copy_(right.cpu())
    return module.state_dict()


@torch.no_grad()
def refine_groupreduce_from_dense(
    dense_weight,
    counts,
    initial_group_ids,
    ranks,
    *,
    max_iters=5,
    move_fraction=0.10,
    min_candidates=1,
    chunk_size=4096,
):
    """Run GroupReduce's reconstruction-based membership refinement.

    Ranks remain attached to group identities, as in the paper.  At each
    iteration we fit weighted-SVD bases, find every token's lowest-error basis,
    move the best 10% (configurable) improving candidates, and refit changed
    groups.  Donor groups are never allowed to shrink below their configured
    rank, which would make the next truncated SVD undefined.
    """
    matrix = torch.as_tensor(dense_weight, dtype=torch.float32)
    weights = torch.as_tensor(
        counts, dtype=torch.float32, device=matrix.device
    )
    group_ids = torch.as_tensor(
        initial_group_ids, dtype=torch.long, device=matrix.device
    ).clone()
    ranks = tuple(int(rank) for rank in ranks)
    vocab_size, embed_dim = matrix.shape
    num_groups = len(ranks)
    if weights.shape != (vocab_size,) or group_ids.shape != (vocab_size,):
        raise ValueError("counts/group ids must have one value per dense row")
    if not 0 < move_fraction <= 1:
        raise ValueError("move_fraction must be in (0, 1]")
    if max_iters < 0 or min_candidates <= 0 or chunk_size <= 0:
        raise ValueError("invalid refinement iteration settings")
    if group_ids.min().item() < 0 or group_ids.max().item() >= num_groups:
        raise ValueError("initial group id is out of range")

    history = []
    for iteration in range(max_iters):
        rights = []
        current_error = torch.empty(vocab_size, device=matrix.device)
        for group, rank in enumerate(ranks):
            token_ids = torch.where(group_ids == group)[0]
            if token_ids.numel() < rank:
                raise ValueError(
                    f"group {group} size {token_ids.numel()} is below rank {rank}"
                )
            _, right = weighted_low_rank_factors(
                matrix[token_ids], weights[token_ids], rank
            )
            rights.append(right)

        best_error = torch.full(
            (vocab_size,), torch.inf, device=matrix.device
        )
        best_group = torch.zeros(
            vocab_size, dtype=torch.long, device=matrix.device
        )
        for start in range(0, vocab_size, chunk_size):
            rows = matrix[start:start + chunk_size]
            row_norm = rows.square().sum(dim=1)
            errors = torch.stack([
                (row_norm - (rows @ right).square().sum(dim=1)).clamp_min(0)
                for right in rights
            ], dim=1)
            chunk_best_error, chunk_best_group = errors.min(dim=1)
            best_error[start:start + rows.size(0)] = chunk_best_error
            best_group[start:start + rows.size(0)] = chunk_best_group
            local_current = group_ids[start:start + rows.size(0)]
            current_error[start:start + rows.size(0)] = errors.gather(
                1, local_current.unsqueeze(1)
            ).squeeze(1)

        candidates = torch.where(
            (best_group != group_ids)
            & (best_error < current_error - 1e-12)
        )[0]
        if candidates.numel() < min_candidates:
            break
        move_limit = max(1, math.ceil(move_fraction * candidates.numel()))
        improvement = current_error[candidates] - best_error[candidates]
        order = torch.argsort(
            improvement, descending=True, stable=True
        )
        sizes = torch.bincount(group_ids, minlength=num_groups)
        moved = 0
        objective_before = float(
            (current_error * weights).sum().item()
        )
        for candidate_position in order.tolist():
            token = candidates[candidate_position]
            source_group = int(group_ids[token].item())
            target_group = int(best_group[token].item())
            if sizes[source_group] <= ranks[source_group]:
                continue
            group_ids[token] = target_group
            sizes[source_group] -= 1
            sizes[target_group] += 1
            moved += 1
            if moved >= move_limit:
                break
        history.append({
            "iteration": iteration,
            "candidates": int(candidates.numel()),
            "moved": moved,
            "weighted_fixed_basis_error_before": objective_before,
        })
        if moved < min_candidates:
            break

    state = initialize_groupreduce_from_dense(
        matrix, weights, group_ids, ranks
    )
    return state, group_ids.cpu(), history


def _pairwise_squared_distances(data, centroids, chunk_size):
    distances = torch.empty(
        data.size(0), centroids.size(0),
        device=data.device, dtype=torch.float32,
    )
    centroid_norm = centroids.square().sum(dim=1)
    for start in range(0, data.size(0), chunk_size):
        chunk = data[start:start + chunk_size]
        distances[start:start + chunk.size(0)] = (
            chunk.square().sum(dim=1, keepdim=True)
            + centroid_norm.unsqueeze(0)
            - 2.0 * chunk @ centroids.T
        ).clamp_min_(0)
    return distances


def _balanced_capacities(num_points, num_codes, device):
    base, remainder = divmod(num_points, num_codes)
    capacities = torch.full(
        (num_codes,), base, dtype=torch.long, device=device
    )
    capacities[:remainder] += 1
    return capacities


@torch.no_grad()
def rebalance_assignments(distances, assignments):
    """Capacity-repair nearest assignments with deterministic greedy moves.

    This is a scalable approximation to the paper's balanced k-means.  It
    enforces exact floor/ceil capacities, but unlike the cited n-by-n Hungarian
    assignment it is not a globally optimal balanced assignment.  Callers must
    record that distinction in experiment metadata.
    """
    num_points, num_codes = distances.shape
    capacities = _balanced_capacities(num_points, num_codes, distances.device)
    assignments = assignments.clone()
    counts = torch.bincount(assignments, minlength=num_codes)

    # Repair all underfull clusters together. Each round gives every surplus
    # point its cheapest currently-underfull destination, then accepts moves in
    # increasing penalty order subject to donor surplus and target capacity.
    # This avoids sorting all surplus points separately for every code.
    while not torch.equal(counts, capacities):
        surplus = counts - capacities
        underfull = torch.where(surplus < 0)[0]
        donor_mask = surplus[assignments] > 0
        candidates = torch.where(donor_mask)[0]
        if underfull.numel() == 0 or candidates.numel() == 0:
            raise RuntimeError("cannot repair balanced cluster capacities")
        current = assignments[candidates]
        alternatives = distances[candidates][:, underfull]
        alternative_cost, alternative_index = alternatives.min(dim=1)
        targets = underfull[alternative_index]
        penalties = alternative_cost - distances[candidates, current]
        order = torch.argsort(penalties, stable=True)
        moved = 0
        for candidate_index in order.tolist():
            point = candidates[candidate_index]
            donor = int(assignments[point].item())
            target = int(targets[candidate_index].item())
            if counts[donor] <= capacities[donor] \
                    or counts[target] >= capacities[target]:
                continue
            assignments[point] = target
            counts[donor] -= 1
            counts[target] += 1
            moved += 1
        if moved == 0:
            raise RuntimeError("balanced capacity repair made no progress")

    if not torch.equal(counts, capacities):
        raise RuntimeError("balanced capacity repair ended with wrong counts")
    return assignments


@torch.no_grad()
def capacity_constrained_kmeans(
    data,
    num_codes,
    *,
    num_iters=20,
    num_restarts=1,
    seed=42,
    chunk_size=4096,
    device=None,
):
    """Deterministic, scalable approximately-balanced Lloyd clustering."""
    source = torch.as_tensor(data, dtype=torch.float32)
    if source.ndim != 2 or source.size(0) < num_codes:
        raise ValueError("data must be a matrix with rows >= num_codes")
    if num_iters <= 0 or num_restarts <= 0 or chunk_size <= 0:
        raise ValueError("iteration/restart/chunk counts must be positive")
    if device is None:
        device = source.device
    work = source.to(device=device, dtype=torch.float32)
    generator = torch.Generator(device="cpu")

    best = None
    for restart in range(num_restarts):
        generator.manual_seed(int(seed) + restart)
        initial_ids = torch.randperm(
            work.size(0), generator=generator
        )[:num_codes].to(work.device)
        centroids = work[initial_ids].clone()
        assignments = None
        for _ in range(num_iters):
            distances = _pairwise_squared_distances(
                work, centroids, chunk_size
            )
            new_assignments = rebalance_assignments(
                distances, distances.argmin(dim=1)
            )
            sums = torch.zeros_like(centroids)
            sums.index_add_(0, new_assignments, work)
            counts = torch.bincount(
                new_assignments, minlength=num_codes
            ).to(dtype=work.dtype)
            new_centroids = sums / counts.unsqueeze(1)
            converged = (
                assignments is not None
                and torch.equal(new_assignments, assignments)
            )
            assignments = new_assignments
            centroids = new_centroids
            if converged:
                break

        distances = _pairwise_squared_distances(work, centroids, chunk_size)
        inertia = distances[
            torch.arange(work.size(0), device=work.device), assignments
        ].sum()
        if best is None or inertia.item() < best[0]:
            best = (
                inertia.item(), centroids.cpu(), assignments.cpu()
            )
    _, centroids, assignments = best
    return centroids, assignments


@torch.no_grad()
def initialize_pvq_from_dense(
    dense_weight,
    shared_dim,
    num_codes,
    assignments,
):
    """Convert a dense table and final fixed codes into compact P-VQ state."""
    dense_weight = torch.as_tensor(dense_weight, dtype=torch.float32)
    module = PVQEmbed(
        dense_weight.size(0),
        dense_weight.size(1),
        shared_dim=shared_dim,
        num_codes=num_codes,
        assignments=assignments,
    )
    module.initialize_from_dense(dense_weight)
    return module.state_dict()
