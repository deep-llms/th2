"""Phase-A bounded routing: configuration, curriculum and exact quota tables."""
from dataclasses import asdict, dataclass
from functools import lru_cache
import math

import torch


@dataclass(frozen=True)
class RoutingConfig:
    n_stem: int = 2
    n_coda: int = 2
    S: int = 8
    mode: str = 'compute_matched'
    window: str = 'absolute'
    kv_mode: str = 'step'
    route_scope: str = 'qo_mlp'
    r: int = 1
    reuse: bool = True
    routing: str = 'learned'
    step_film: bool = True
    visit_scale: str = 'none'
    gate: str = 'st'
    default_prior: float = 1.0
    init_std_scale: float = 0.1
    balance_rate: float = 1e-3
    z_loss_coef: float = 1e-3
    explore_eps_max: float = 0.05
    curriculum: str = 'B'
    soft_phase_frac: float = 0.15
    explore_end_frac: float = 0.5
    widen_every_frac: float = 0.1
    seed: int = 0
    expert_backend: str = 'reference'
    logit_cap: float | None = None
    select_on: str = 'logits'
    placement_film: bool = False
    placement_film_scope: str = 'both'

    def __post_init__(self):
        for key in ('S', 'n_stem', 'n_coda', 'r', 'seed'):
            x = getattr(self, key)
            if type(x) is not int or x < (1 if key == 'S' else 0):
                raise ValueError(f'Invalid routing {key}: {x}')
        if (self.mode, self.window, self.kv_mode, self.route_scope, self.visit_scale) != (
                'compute_matched', 'absolute', 'step', 'qo_mlp', 'none'):
            raise ValueError('Phase A supports compute_matched/absolute/step/qo_mlp/visit_scale=none only')
        if self.routing not in ('fixed', 'learned', 'random') or self.gate not in ('st', 'scale'):
            raise ValueError('Unsupported routing/gate')
        if self.curriculum not in ('A', 'B', 'none'):
            raise ValueError('curriculum must be A, B or none')
        if self.expert_backend not in ('reference', 'grouped'):
            raise ValueError('expert_backend must be reference or grouped')
        if self.select_on not in ('logits', 'probs'):
            raise ValueError('select_on must be logits or probs')
        if self.logit_cap is not None and (isinstance(self.logit_cap, bool)
                or not math.isfinite(self.logit_cap) or self.logit_cap <= 0):
            raise ValueError('logit_cap must be positive and finite, or null')
        if type(self.placement_film) is not bool or self.placement_film_scope not in ('both', 'query', 'mlp'):
            raise ValueError('Invalid placement FiLM configuration')
        if not self.placement_film and self.placement_film_scope != 'both':
            raise ValueError('placement_film_scope requires placement_film')
        if self.routing != 'learned' and self.curriculum != 'none':
            raise ValueError('Fixed/random controls require curriculum=none')
        if not 0 < self.soft_phase_frac < self.explore_end_frac <= 1:
            raise ValueError('Require 0 < soft_phase_frac < explore_end_frac <= 1')
        if not 0 < self.widen_every_frac <= 1 or not 0 <= self.explore_eps_max <= 1:
            raise ValueError('Invalid widening/exploration fraction')
        for key in ('default_prior', 'init_std_scale', 'balance_rate', 'z_loss_coef'):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid routing {key}')

    def to_dict(self):
        return asdict(self)

    def validate_schedule(self, max_steps, warmup_steps):
        """Keep the existing single horizon; reject overlapping warmup/transition."""
        if self.routing != 'learned' or self.curriculum == 'none':
            return
        fraction = self.widen_every_frac if self.curriculum == 'A' else self.soft_phase_frac
        boundary = math.ceil(fraction * max_steps)
        if warmup_steps >= boundary:
            raise ValueError(f'LR warmup ({warmup_steps}) must end before curriculum '
                             f'{self.curriculum} first boundary ({boundary})')

    def phase(self, progress, training, *, evaluation_policy='target'):
        """Progress is completed optimizer steps / full schedule (equal fixed batches)."""
        if evaluation_policy not in ('target', 'curriculum'):
            raise ValueError('evaluation_policy must be target or curriculum')
        if not training and evaluation_policy == 'curriculum':
            reach, soft, _ = self.phase(progress, True)
            return reach, soft, 0.  # Never turn exploration on for validation.
        if not training or self.routing != 'learned' or self.curriculum == 'none':
            return self.r, False, 0.
        if self.curriculum == 'B':
            soft = progress < self.soft_phase_frac
            eps = 0. if soft else self.explore_eps_max * max(
                0., (self.explore_end_frac - progress) / (self.explore_end_frac - self.soft_phase_frac))
            return self.r, soft, eps
        # Widen at 10%, 20%, ...; exploration resets only on an actual widening.
        stage = int(math.floor(progress / self.widen_every_frac + 1e-10))
        reach = min(self.r, stage)
        eps = 0.
        if 0 < stage <= self.r:
            eps = self.explore_eps_max * max(0., 1. - (progress - stage * self.widen_every_frac) / self.widen_every_frac)
        return reach, False, eps


def quota_table(steps, reach):
    """Reachable feasible states only; no greedy pruning and no token-side search."""
    candidates = [tuple(range(max(0, s-reach), min(steps, s+reach+1))) for s in range(steps)]

    @lru_cache(None)
    def feasible(s, remaining):
        if s == steps:
            return remaining == 0
        return any(remaining & (1 << m) and feasible(s+1, remaining ^ (1 << m)) for m in candidates[s])

    states = [(0, (1 << steps)-1)]
    indices = {states[0]: 0}
    allowed, transitions = [], []
    for s, remaining in states:
        row, nxt = [False]*steps, [-1]*steps
        if s < steps:
            for m in candidates[s]:
                if remaining & (1 << m) and feasible(s+1, remaining ^ (1 << m)):
                    state = (s+1, remaining ^ (1 << m))
                    if state not in indices:
                        indices[state] = len(states)
                        states.append(state)
                    row[m], nxt[m] = True, indices[state]
        allowed.append(row)
        transitions.append(nxt)
    return torch.tensor(allowed, dtype=torch.bool), torch.tensor(transitions, dtype=torch.long)
