"""Nested frequency-ladder embedding with an exactly tied output head.

Every token owns a base low-rank representation.  Successively smaller,
frequency-ranked vocabulary subsets receive additional factor rows and basis
directions.  Membership is static in the Phase-1 experiments and is stored in
the state dict, making checkpoints independent of the original frequency file.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .compression_init import frequency_rank_order


def nested_ladder_parameter_count(vocab_size, embed_dim, ranks, populations):
    """Return the exact number of trainable embedding parameters."""
    ranks = tuple(int(rank) for rank in ranks)
    populations = tuple(int(size) for size in populations)
    if len(ranks) != len(populations):
        raise ValueError("ranks and populations must have equal length")
    return int(embed_dim) + sum(
        rank * (size + int(embed_dim))
        for rank, size in zip(ranks, populations)
    )


class NestedLadderEmbed(nn.Module):
    """Additive low-rank tiers assigned by deterministic token frequency.

    For token ``i`` the effective embedding is::

        e_i = bias + sum_t Z_t[slot_t(i)] @ W_t.T

    Tier 1 contains the full vocabulary.  Every later tier is a nested prefix
    of the frequency ranking, with token-id as the deterministic tie-breaker.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        tier_ranks,
        tier_populations,
        *,
        counts=None,
        member_ids=None,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.tier_ranks = tuple(int(rank) for rank in tier_ranks)
        self.tier_populations = tuple(
            int(size) for size in tier_populations
        )
        self.num_tiers = len(self.tier_ranks)
        # Short aliases match the method notation and make head/debug code
        # straightforward without changing the repository-wide names.
        self.V = self.vocab_size
        self.d = self.embed_dim
        self.T = self.num_tiers
        self._validate_hyperparameters()

        upper_members = self._resolve_members(counts, member_ids)
        for tier, ids in enumerate(upper_members, start=2):
            self.register_buffer(f"member_ids_{tier}", ids, persistent=True)
            slot = torch.full((self.vocab_size,), -1, dtype=torch.long)
            slot[ids] = torch.arange(ids.numel(), dtype=torch.long)
            self.register_buffer(
                f"member_slot_{tier}", slot, persistent=True
            )

        self.Z = nn.ParameterList([
            nn.Parameter(torch.empty(size, rank))
            for size, rank in zip(
                self.tier_populations, self.tier_ranks
            )
        ])
        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(self.embed_dim, rank))
            for rank in self.tier_ranks
        ])
        self.bias = nn.Parameter(torch.zeros(self.embed_dim))
        self.reset_parameters()
        self.validate_structure()
        self.register_load_state_dict_post_hook(self._validate_after_load)

        # Transient observed-batch diagnostics; never serialized.
        self._step_metrics = None

    def _validate_hyperparameters(self):
        if self.vocab_size <= 0 or self.embed_dim <= 0:
            raise ValueError("vocab_size and embed_dim must be positive")
        if self.num_tiers == 0:
            raise ValueError("at least one tier is required")
        if len(self.tier_populations) != self.num_tiers:
            raise ValueError("tier_ranks and tier_populations must match")
        if any(not 0 < rank <= self.embed_dim for rank in self.tier_ranks):
            raise ValueError("every tier rank must be in [1, embed_dim]")
        if self.tier_populations[0] != self.vocab_size:
            raise ValueError("tier 1 population must equal vocab_size")
        if any(size <= 0 for size in self.tier_populations):
            raise ValueError("tier populations must be positive")
        if any(
            current >= previous
            for previous, current in zip(
                self.tier_populations, self.tier_populations[1:]
            )
        ):
            raise ValueError("tier populations must be strictly decreasing")

    @classmethod
    def structure_from_state(cls, state):
        """Validate and extract checkpoint-authoritative ladder structure."""
        if not isinstance(state, dict):
            raise TypeError("Nested Ladder state must be a dictionary")
        keys = set(state)
        if not {"Z.0", "W.0", "bias"}.issubset(keys):
            raise ValueError(
                "Nested Ladder state requires Z.0, W.0, and bias"
            )

        def indexed_keys(prefix):
            found = []
            for key in keys:
                if not key.startswith(prefix + "."):
                    continue
                suffix = key.removeprefix(prefix + ".")
                if not suffix.isdigit():
                    raise ValueError(
                        f"Nested Ladder state has invalid key {key!r}"
                    )
                found.append((int(suffix), key))
            found.sort()
            expected = list(range(len(found)))
            if [index for index, _ in found] != expected:
                raise ValueError(
                    f"Nested Ladder {prefix} tiers are not contiguous"
                )
            return [key for _, key in found]

        z_keys = indexed_keys("Z")
        w_keys = indexed_keys("W")
        if len(z_keys) != len(w_keys):
            raise ValueError("Nested Ladder state has mismatched Z/W tiers")
        if state["bias"].ndim != 1:
            raise ValueError("Nested Ladder bias must be one-dimensional")
        embed_dim = int(state["bias"].numel())
        ranks = []
        populations = []
        for index, (z_key, w_key) in enumerate(zip(z_keys, w_keys)):
            z, w = state[z_key], state[w_key]
            if z.ndim != 2 or w.ndim != 2:
                raise ValueError(
                    f"Nested Ladder tier {index + 1} factors must be matrices"
                )
            if w.shape[0] != embed_dim or z.shape[1] != w.shape[1]:
                raise ValueError(
                    f"Nested Ladder tier {index + 1} has incompatible Z/W shapes"
                )
            populations.append(int(z.shape[0]))
            ranks.append(int(w.shape[1]))

        member_ids = []
        vocab_size = populations[0]
        for tier in range(2, len(ranks) + 1):
            ids_key = f"member_ids_{tier}"
            slot_key = f"member_slot_{tier}"
            missing = {ids_key, slot_key} - keys
            if missing:
                raise ValueError(
                    f"Nested Ladder state is missing tier-{tier} buffers: "
                    f"{sorted(missing)}"
                )
            ids, slots = state[ids_key], state[slot_key]
            if ids.dtype != torch.long or slots.dtype != torch.long:
                raise ValueError(
                    f"Nested Ladder tier-{tier} membership buffers must be int64"
                )
            if ids.shape != (populations[tier - 1],):
                raise ValueError(
                    f"Nested Ladder {ids_key} has incompatible shape"
                )
            if slots.shape != (vocab_size,):
                raise ValueError(
                    f"Nested Ladder {slot_key} has incompatible shape"
                )
            member_ids.append(ids)

        return {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "tier_ranks": tuple(ranks),
            "tier_populations": tuple(populations),
            "member_ids": tuple(member_ids),
        }

    def _resolve_members(self, counts, member_ids):
        expected = self.num_tiers - 1
        if member_ids is not None:
            members = list(member_ids)
            if len(members) != expected:
                raise ValueError(
                    f"expected {expected} upper-tier member lists, got "
                    f"{len(members)}"
                )
            return [
                torch.sort(torch.as_tensor(ids, dtype=torch.long).cpu()).values
                for ids in members
            ]

        if expected == 0:
            return []
        if counts is None:
            raise ValueError(
                "counts are required when upper-tier membership is not supplied"
            )
        counts = torch.as_tensor(counts, dtype=torch.float64).cpu()
        if counts.shape != (self.vocab_size,):
            raise ValueError(
                f"counts must have shape ({self.vocab_size},), got "
                f"{tuple(counts.shape)}"
            )
        if not torch.isfinite(counts).all() or torch.any(counts < 0):
            raise ValueError("counts must be finite and nonnegative")
        # Stable descending sort makes the original token-id order the exact
        # tie-breaker. Buffers themselves are sorted for lookup efficiency.
        order = frequency_rank_order(counts)
        return [
            torch.sort(order[:size]).values
            for size in self.tier_populations[1:]
        ]

    def reset_parameters(self):
        for parameter in (*self.Z, *self.W):
            nn.init.normal_(parameter, std=0.02)
        nn.init.zeros_(self.bias)

    @property
    def parameter_count(self):
        return nested_ladder_parameter_count(
            self.vocab_size,
            self.embed_dim,
            self.tier_ranks,
            self.tier_populations,
        )

    @property
    def member_ids(self):
        return [
            getattr(self, f"member_ids_{tier}")
            for tier in range(2, self.num_tiers + 1)
        ]

    @property
    def member_slots(self):
        return [
            getattr(self, f"member_slot_{tier}")
            for tier in range(2, self.num_tiers + 1)
        ]

    def validate_structure(self):
        """Validate exact inverse maps, sizes, bounds, and tier nesting."""
        previous = None
        for tier in range(2, self.num_tiers + 1):
            ids = getattr(self, f"member_ids_{tier}")
            slot = getattr(self, f"member_slot_{tier}")
            expected_size = self.tier_populations[tier - 1]
            if ids.dtype != torch.long or slot.dtype != torch.long:
                raise ValueError("member ids and slots must use torch.long")
            if ids.shape != (expected_size,):
                raise ValueError(
                    f"tier {tier} member ids have shape {tuple(ids.shape)}; "
                    f"expected ({expected_size},)"
                )
            if slot.shape != (self.vocab_size,):
                raise ValueError(f"tier {tier} member_slot has invalid shape")
            if torch.any(ids < 0) or torch.any(ids >= self.vocab_size):
                raise ValueError(f"tier {tier} has out-of-range token ids")
            if ids.numel() > 1 and torch.any(ids[1:] <= ids[:-1]):
                raise ValueError(f"tier {tier} member ids must be sorted unique")
            expected_slot = torch.full_like(slot, -1)
            expected_slot[ids] = torch.arange(
                ids.numel(), dtype=torch.long, device=ids.device
            )
            if not torch.equal(slot, expected_slot):
                raise ValueError(f"tier {tier} member_slot is not the exact inverse")
            if previous is not None and torch.any(previous[ids] < 0):
                raise ValueError(f"tier {tier} is not nested inside tier {tier - 1}")
            previous = slot
        return True

    @staticmethod
    def _validate_after_load(module, incompatible_keys):
        module.validate_structure()

    def materialize(self, token_ids=None):
        """Materialize effective embeddings for selected tokens (or all V)."""
        if token_ids is None:
            token_ids = torch.arange(self.vocab_size, device=self.bias.device)
        token_ids = torch.as_tensor(
            token_ids, dtype=torch.long, device=self.bias.device
        )
        flat_ids = token_ids.reshape(-1)
        result = self.Z[0][flat_ids] @ self.W[0].T + self.bias
        for index in range(1, self.num_tiers):
            slot = getattr(self, f"member_slot_{index + 1}")[flat_ids]
            positions = torch.nonzero(slot >= 0, as_tuple=False).squeeze(-1)
            rows = slot[positions]
            # Executing the empty matmul/index_add is intentional: every tier
            # remains in autograd even when a rank's tail batch has no members.
            contribution = self.Z[index][rows] @ self.W[index].T
            result = result.index_add(0, positions, contribution)
        return result.view(*token_ids.shape, self.embed_dim)

    def forward(self, input_ids, doc_mask=None):
        embeddings = self.materialize(input_ids)
        with torch.no_grad():
            flat_ids = input_ids.reshape(-1)
            depths = torch.ones_like(flat_ids)
            for tier in range(2, self.num_tiers + 1):
                slot = getattr(self, f"member_slot_{tier}")[flat_ids]
                depths[slot >= 0] = tier
            norms = embeddings.detach().reshape(
                -1, self.embed_dim
            ).float().norm(dim=-1)
            depth_indices = depths - 1
            sums = torch.zeros(
                self.num_tiers, device=norms.device, dtype=norms.dtype
            ).index_add_(0, depth_indices, norms)
            counts = torch.bincount(
                depth_indices, minlength=self.num_tiers
            )
            self._step_metrics = {
                f"nested_depth_{depth}_embedding_norm": (
                    sums[depth - 1], counts[depth - 1]
                )
                for depth in range(1, self.num_tiers + 1)
            }
        return embeddings, None

    def pop_step_metrics(self):
        """Return and clear device-side ``metric: (sum, count)`` values."""
        metrics = self._step_metrics
        self._step_metrics = None
        return metrics

    @torch.no_grad()
    def reassign_zero_row(self, tier, old_token, new_token):
        """Move one zero contribution row without changing the effective table.

        This is the structural primitive needed by a future grow/reallocation
        phase. It deliberately rejects nonzero rows; optimizer-state migration
        and the learned policy are outside the Phase-1 implementation.
        """
        tier = int(tier)
        if not 2 <= tier <= self.num_tiers:
            raise ValueError("tier must name an upper tier (2..T)")
        ids = getattr(self, f"member_ids_{tier}")
        slot = getattr(self, f"member_slot_{tier}")
        old_token, new_token = int(old_token), int(new_token)
        old_row = int(slot[old_token].item())
        if old_row < 0 or int(slot[new_token].item()) >= 0:
            raise ValueError("old_token must be a member and new_token a non-member")
        if tier > 2 and int(getattr(
            self, f"member_slot_{tier - 1}"
        )[new_token].item()) < 0:
            raise ValueError("new_token must belong to the preceding tier")
        if tier < self.num_tiers and int(getattr(
            self, f"member_slot_{tier + 1}"
        )[old_token].item()) >= 0:
            raise ValueError("cannot remove a token retained by a deeper tier")
        if torch.count_nonzero(self.Z[tier - 1][old_row]).item() != 0:
            raise ValueError("the reassigned factor row must be exactly zero")

        replacement = ids.clone()
        replacement[old_row] = new_token
        sorted_ids, permutation = torch.sort(replacement)
        self.Z[tier - 1].data.copy_(self.Z[tier - 1].data[permutation])
        ids.copy_(sorted_ids)
        slot.fill_(-1)
        slot[ids] = torch.arange(ids.numel(), device=ids.device)
        self.validate_structure()
