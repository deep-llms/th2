"""Causal single-edit policy at a fixed depth; frozen LLM, held factual gate."""
import torch
from torch import nn
from .window_cf import Intervention
from .window_order import prior_features, bounds


def decisions(length, window, stride):
    return [(d, j, a, b) for d in range(4)
            for j, (a, b) in enumerate(bounds(length, window, (d * stride) % window))]


def first_edit(predicted, factual, threshold):
    """Select earliest positive improvement. No future-window statistic is used."""
    if threshold < 0 or predicted.shape[:-1] != factual.shape:
        raise ValueError('Invalid policy inputs')
    value, candidate = predicted.min(-1)
    old = predicted.gather(-1, factual[..., None]).squeeze(-1)
    eligible = old - value > threshold
    first = eligible & (eligible.long().cumsum(1) == 1)
    return torch.where(first, candidate, factual), first


class CostHead(nn.Module):
    def __init__(self, dimension, hidden=128):
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.net = nn.Sequential(nn.Linear(dimension + 6, hidden), nn.GELU(), nn.Linear(hidden, 4))

    def forward(self, features, metadata):
        # Four depth coordinates and normalized window start/end.
        return self.net(torch.cat((self.norm(features.float()), metadata.float()), -1)) / 100


def metadata(length, window, stride, device='cpu'):
    rows = []
    for d, j, a, b in decisions(length, window, stride):
        rows.append([float(z == d) for z in range(4)] + [a / length, b / length])
    return torch.tensor(rows, device=device)


class Policy(Intervention):
    """One selected depth makes the edit budget causal across token positions.

    A budget shared across different depths is deliberately unsupported: choices
    at late tokens of an early depth must not control early tokens at later depths.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_head = None
        self.constant_cost = None
        self.policy_depth = None
        self.threshold = 0.
        self.features = []
        self.factual_routes = []
        self.edit_masks = []

    def hidden(self, x, path=None):
        if self.training:
            raise ValueError('Policy body is frozen and evaluation-only')
        self.features = []; self.factual_routes = []; self.edit_masks = []
        return super().hidden(x, path)

    def choose(self, h, x, spans, depth):
        choices, gates = super().choose(h, x, spans, depth)
        features = prior_features(h, spans)
        self.features.append(features.detach())
        self.factual_routes.append(choices.detach())
        mask = torch.zeros_like(choices, dtype=torch.bool)
        if self.policy_depth == depth:
            offset = sum(len(bounds(h.shape[1], self.window, (d*self.stride) % self.window)) for d in range(depth))
            if self.constant_cost is not None:
                predicted = self.constant_cost[offset:offset+len(spans)].to(h.device)[None].expand(h.shape[0], -1, -1)
            elif self.cost_head is not None:
                meta = metadata(h.shape[1], self.window, self.stride, h.device)[offset:offset+len(spans)]
                predicted = self.cost_head(features, meta[None].expand(h.shape[0], -1, -1))
            else:
                raise ValueError('Policy has no cost predictor')
            choices, mask = first_edit(predicted, choices, self.threshold)
        self.edit_masks.append(mask.detach())
        return choices, gates


def cached_policy(predicted, factual, costs, original, depth, threshold, layout):
    """Evaluate a label-free policy from previously enumerated single-edit costs."""
    if depth is None:
        return original.clone(), torch.full_like(factual[:, 0], -1)
    indices = [i for i, row in enumerate(layout) if row[0] == depth]
    choices, edits = first_edit(predicted[:, indices], factual[:, indices], threshold)
    result = original.clone(); action = torch.full_like(factual[:, 0], -1)
    for local, index in enumerate(indices):
        which = index * 4 + choices[:, local]
        mask = edits[:, local]
        value = costs.gather(1, which[:, None]).squeeze(1)
        result = torch.where(mask, value, result)
        action = torch.where(mask, which, action)
    return result, action
