"""Nonlinear and deep factorization controls for tied token interfaces.

The classes here intentionally separate three hypotheses:

* :class:`FunnelingEmbed` is the from-scratch architectural form of Distilled
  Embedding's nonlinear factorization.  Its nonlinearity does not increase the
  feature width, so the centered effective table remains rank-limited.
* :class:`DeFINEEmbed` is a fixed-tokenizer Qwen adaptation of DeFINE's
  map-expand-reduce input transform.  Its hierarchical group transforms use
  independent matrices per group, group-wise LayerNorm, and GELU, following
  the authors' released DeFINE/DeLighT implementation.  It ties the
  low-dimensional token table to a low-dimensional flat classifier, as
  DeFINE does, rather than tying the expanded input table.
* :class:`RankLiftEmbed` expands each private token code to a wider nonlinear
  feature vector and uses that same expanded effective table for input and
  output.  Its exact output algebra is implemented in ``tied_head.py``.
* :class:`TieredRankLiftEmbed` applies that expansion inside static vocabulary
  tiers.  It preserves GroupReduce's disjoint capacity allocation while
  decoupling stored token-code width from classifier-feature width in the
  lower-capacity tiers.

All modules return ``(embedding, None)`` to match the project's common
embedding interface.  They own every trainable output-side projection so the
tied heads can keep non-registering references and never duplicate parameters
under ``lm_head``.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_common(vocab_size: int, embed_dim: int, code_dim: int) -> None:
    if vocab_size <= 0 or embed_dim <= 0 or code_dim <= 0:
        raise ValueError("vocab_size, embed_dim, and code_dim must be positive")


def _parameter_free_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    # Accumulating the second moment in FP32 avoids BF16 overflow/underflow and
    # matches the numerical policy used by modern RMSNorm implementations.
    variance = x.float().square().mean(dim=-1, keepdim=True)
    scale = torch.rsqrt(variance + eps).to(dtype=x.dtype)
    return x * scale


class RankLiftEmbed(nn.Module):
    """Nonlinearly lift private token codes before a shared projection.

    For code ``z_i`` the feature and effective embedding are

        u_i = RMSNorm(z_i)
        g_i = SiLU(A(u_i)) * B(u_i)
        f_i = concat(z_i, g_i)
        e_i = P(f_i)

    ``P`` includes the repository-standard global embedding bias.  The lift
    linears include biases deliberately; they are custom learned affine layers,
    not bias-free Qwen projections.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        code_dim: int = 124,
        lift_dim: int = 336,
        rms_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        _validate_common(vocab_size, embed_dim, code_dim)
        if lift_dim <= 0:
            raise ValueError("lift_dim must be positive")
        if not math.isfinite(rms_eps) or rms_eps <= 0:
            raise ValueError("rms_eps must be finite and positive")

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.code_dim = int(code_dim)
        self.lift_dim = int(lift_dim)
        self.feature_dim = self.code_dim + self.lift_dim
        self.rms_eps = float(rms_eps)
        # Preserve the exact Python-float hyperparameter in embedding.pt.
        # Integer buffers are not rounded by model.to(dtype=bf16), unlike a
        # floating buffer, while configless loading can reconstruct the bits.
        rms_eps_bits = struct.unpack("<q", struct.pack("<d", self.rms_eps))[0]
        self.register_buffer(
            "rms_eps_bits",
            torch.tensor(rms_eps_bits, dtype=torch.int64),
            persistent=True,
        )

        self.token_codes = nn.Parameter(
            torch.empty(self.vocab_size, self.code_dim)
        )
        self.lift_a = nn.Linear(self.code_dim, self.lift_dim, bias=True)
        self.lift_b = nn.Linear(self.code_dim, self.lift_dim, bias=True)
        self.projection = nn.Linear(
            self.feature_dim, self.embed_dim, bias=True
        )
        self.reset_parameters()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # ``rms_eps`` is a Python float used directly by forward(), while its
        # exact bits are persisted as an integer buffer for configless loading.
        # A plain Module.load_state_dict() would otherwise be able to overwrite
        # only the buffer and silently leave forward() using the constructor's
        # different epsilon.  Production/configless loaders construct with the
        # saved value; reject inconsistent direct loads instead of accepting a
        # checkpoint whose state and behavior disagree.
        bits_key = prefix + "rms_eps_bits"
        if bits_key in state_dict:
            saved_bits = int(state_dict[bits_key].item())
            saved_eps = struct.unpack("<d", struct.pack("<q", saved_bits))[0]
            if saved_eps != self.rms_eps:
                error_msgs.append(
                    f"{bits_key} encodes rms_eps={saved_eps!r}, but the "
                    f"module was constructed with rms_eps={self.rms_eps!r}"
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_codes, mean=0.0, std=0.02)
        # Match the project's Qwen/LowRank 0.02 initialization.  Xavier on an
        # RMS-normalized code would make each lift branch O(1); their product
        # would then dominate the raw 0.02-scale private codes and produce an
        # excessively large effective embedding table at step zero.
        nn.init.normal_(self.lift_a.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lift_b.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lift_a.bias)
        nn.init.zeros_(self.lift_b.bias)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def features_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        normalized = _parameter_free_rms_norm(codes, self.rms_eps)
        lifted = F.silu(self.lift_a(normalized)) * self.lift_b(normalized)
        return torch.cat((codes, lifted), dim=-1)

    def materialize_features(self) -> torch.Tensor:
        """Return the current ``V x feature_dim`` table with live autograd."""
        return self.features_from_codes(self.token_codes)

    def materialize_effective_table(self) -> torch.Tensor:
        """Reference/debug path; the production head must not call this."""
        return self.projection(self.materialize_features())

    def forward(self, input_ids, doc_mask=None):
        # F.embedding validates indices in the CUDA kernel without a per-step
        # ``min().item()`` host synchronization.
        codes = F.embedding(input_ids, self.token_codes)
        features = self.features_from_codes(codes)
        return self.projection(features), None


class TieredRankLiftEmbed(nn.Module):
    """Disjoint frequency tiers with optional nonlinear rank expansion.

    Group ``g`` stores a private code ``z_i`` of width ``c_g``.  When its lift
    width ``q_g`` is non-zero, its tied feature and effective embedding are

        u_i = RMSNorm(z_i)
        f_i = concat(z_i, SiLU(A_g(u_i)) * B_g(u_i))
        e_i = f_i @ R_g.T

    where ``R_g`` has shape ``(embed_dim, c_g + q_g)``.  A zero lift recovers
    the ordinary GroupReduce block exactly: ``f_i = z_i``.  Group membership
    is static and persistent, so checkpoint loading never recomputes it from
    a possibly changed external importance file.

    The lift linears include biases deliberately.  They are custom affine
    layers (not Qwen projections), and the registered design budget includes
    both biases.  The final block projection remains bias-free to match the
    GroupReduce control and keep the comparison isolated to nonlinear width.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        code_dims: Sequence[int],
        lift_dims: Sequence[int],
        group_ids=None,
        rms_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or embed_dim <= 0:
            raise ValueError("vocab_size and embed_dim must be positive")
        code_dims = tuple(int(value) for value in code_dims)
        lift_dims = tuple(int(value) for value in lift_dims)
        if not code_dims or len(code_dims) != len(lift_dims):
            raise ValueError(
                "code_dims and lift_dims must be non-empty and have equal length"
            )
        if any(value <= 0 or value > embed_dim for value in code_dims):
            raise ValueError(
                f"code dimensions must be in [1, {embed_dim}], got {code_dims}"
            )
        if any(value < 0 for value in lift_dims):
            raise ValueError(f"lift dimensions must be nonnegative, got {lift_dims}")
        if len(code_dims) > vocab_size:
            raise ValueError("number of tiers cannot exceed vocabulary size")
        if not math.isfinite(rms_eps) or rms_eps <= 0:
            raise ValueError("rms_eps must be finite and positive")

        num_groups = len(code_dims)
        if group_ids is None:
            base, remainder = divmod(vocab_size, num_groups)
            sizes = [base + (group < remainder) for group in range(num_groups)]
            group_ids = torch.repeat_interleave(
                torch.arange(num_groups, dtype=torch.long),
                torch.tensor(sizes, dtype=torch.long),
            )
        else:
            group_ids = torch.as_tensor(
                group_ids, dtype=torch.long
            ).detach().clone()
        if group_ids.ndim != 1 or group_ids.numel() != vocab_size:
            raise ValueError(
                f"group_ids must have shape ({vocab_size},), got "
                f"{tuple(group_ids.shape)}"
            )
        if group_ids.min().item() < 0 or group_ids.max().item() >= num_groups:
            raise ValueError(f"group_ids must be in [0, {num_groups - 1}]")
        group_sizes = torch.bincount(group_ids, minlength=num_groups)
        if torch.any(group_sizes == 0):
            missing = torch.where(group_sizes == 0)[0].tolist()
            raise ValueError(f"every Tiered RankLift group must be used; missing {missing}")

        token_ids_by_group = torch.cat([
            torch.where(group_ids == group)[0] for group in range(num_groups)
        ])
        inverse_grouped_order = torch.empty(vocab_size, dtype=torch.long)
        inverse_grouped_order[token_ids_by_group] = torch.arange(
            vocab_size, dtype=torch.long
        )
        offsets = torch.empty(vocab_size, dtype=torch.long)
        start = 0
        for size in group_sizes.tolist():
            token_ids = token_ids_by_group[start:start + size]
            offsets[token_ids] = torch.arange(size, dtype=torch.long)
            start += size

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.num_groups = int(num_groups)
        self.code_dims = code_dims
        self.lift_dims = lift_dims
        self.feature_dims = tuple(
            code_dim + lift_dim
            for code_dim, lift_dim in zip(code_dims, lift_dims)
        )
        self.group_sizes = tuple(int(size) for size in group_sizes.tolist())
        self.rms_eps = float(rms_eps)

        self.token_codes = nn.ParameterList([
            nn.Parameter(torch.empty(size, code_dim))
            for size, code_dim in zip(self.group_sizes, self.code_dims)
        ])
        self.lift_a = nn.ModuleList([
            nn.Linear(code_dim, lift_dim, bias=True)
            if lift_dim else nn.Identity()
            for code_dim, lift_dim in zip(self.code_dims, self.lift_dims)
        ])
        self.lift_b = nn.ModuleList([
            nn.Linear(code_dim, lift_dim, bias=True)
            if lift_dim else nn.Identity()
            for code_dim, lift_dim in zip(self.code_dims, self.lift_dims)
        ])
        self.right_factors = nn.ParameterList([
            nn.Parameter(torch.empty(embed_dim, feature_dim))
            for feature_dim in self.feature_dims
        ])

        self.register_buffer("group_ids", group_ids, persistent=True)
        self.register_buffer("group_offsets", offsets, persistent=True)
        self.register_buffer(
            "token_ids_by_group", token_ids_by_group, persistent=True
        )
        self.register_buffer(
            "inverse_grouped_order", inverse_grouped_order, persistent=True
        )
        self.register_buffer(
            "tier_code_dims",
            torch.tensor(self.code_dims, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "tier_lift_dims",
            torch.tensor(self.lift_dims, dtype=torch.long),
            persistent=True,
        )
        rms_eps_bits = struct.unpack("<q", struct.pack("<d", self.rms_eps))[0]
        self.register_buffer(
            "tier_rms_eps_bits",
            torch.tensor(rms_eps_bits, dtype=torch.int64),
            persistent=True,
        )
        self.reset_parameters()
        self.register_load_state_dict_post_hook(
            lambda module, incompatible: module.validate_structure()
        )

    @classmethod
    def structure_from_state(cls, state):
        """Recover persistent topology without relying on train_config.json."""
        required = {
            "group_ids", "tier_code_dims", "tier_lift_dims",
            "tier_rms_eps_bits",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(
                "Tiered RankLift state is missing structural tensors: "
                f"{sorted(missing)}"
            )
        group_ids = torch.as_tensor(state["group_ids"], dtype=torch.long).cpu()
        code_dims = tuple(
            int(value) for value in state["tier_code_dims"].cpu().tolist()
        )
        lift_dims = tuple(
            int(value) for value in state["tier_lift_dims"].cpu().tolist()
        )
        if not code_dims or len(code_dims) != len(lift_dims):
            raise ValueError("invalid Tiered RankLift dimension metadata")
        right_keys = [f"right_factors.{group}" for group in range(len(code_dims))]
        code_keys = [f"token_codes.{group}" for group in range(len(code_dims))]
        missing_parameters = [
            key for key in right_keys + code_keys if key not in state
        ]
        if missing_parameters:
            raise ValueError(
                "Tiered RankLift state is missing parameters: "
                f"{missing_parameters}"
            )
        embed_dims = {int(state[key].shape[0]) for key in right_keys}
        if len(embed_dims) != 1:
            raise ValueError("Tiered RankLift right factors disagree on embed_dim")
        rms_bits = int(state["tier_rms_eps_bits"].item())
        rms_eps = struct.unpack("<d", struct.pack("<q", rms_bits))[0]
        return {
            "vocab_size": int(group_ids.numel()),
            "embed_dim": embed_dims.pop(),
            "code_dims": code_dims,
            "lift_dims": lift_dims,
            "group_ids": group_ids,
            "group_sizes": tuple(
                int(value) for value in torch.bincount(
                    group_ids, minlength=len(code_dims)
                ).tolist()
            ),
            "rms_eps": rms_eps,
        }

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        bits_key = prefix + "tier_rms_eps_bits"
        if bits_key in state_dict:
            saved_bits = int(state_dict[bits_key].item())
            saved_eps = struct.unpack("<d", struct.pack("<q", saved_bits))[0]
            if saved_eps != self.rms_eps:
                error_msgs.append(
                    f"{bits_key} encodes rms_eps={saved_eps!r}, but the "
                    f"module was constructed with rms_eps={self.rms_eps!r}"
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def reset_parameters(self) -> None:
        for codes in self.token_codes:
            nn.init.normal_(codes, mean=0.0, std=0.02)
        for group, lift_dim in enumerate(self.lift_dims):
            if lift_dim:
                nn.init.normal_(self.lift_a[group].weight, mean=0.0, std=0.02)
                nn.init.normal_(self.lift_b[group].weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.lift_a[group].bias)
                nn.init.zeros_(self.lift_b[group].bias)
        for factor in self.right_factors:
            nn.init.normal_(factor, mean=0.0, std=0.02)

    def validate_structure(self) -> None:
        if tuple(self.tier_code_dims.cpu().tolist()) != self.code_dims:
            raise ValueError("Tiered RankLift code-dimension metadata is inconsistent")
        if tuple(self.tier_lift_dims.cpu().tolist()) != self.lift_dims:
            raise ValueError("Tiered RankLift lift-dimension metadata is inconsistent")
        expected_ids = torch.cat([
            torch.where(self.group_ids == group)[0]
            for group in range(self.num_groups)
        ])
        if not torch.equal(self.token_ids_by_group, expected_ids):
            raise ValueError("Tiered RankLift grouped token order is inconsistent")
        expected_inverse = torch.empty_like(self.inverse_grouped_order)
        expected_inverse[expected_ids] = torch.arange(
            self.vocab_size, device=expected_ids.device
        )
        if not torch.equal(self.inverse_grouped_order, expected_inverse):
            raise ValueError("Tiered RankLift inverse token order is inconsistent")
        expected_offsets = torch.empty_like(self.group_offsets)
        start = 0
        for group, size in enumerate(self.group_sizes):
            ids = expected_ids[start:start + size]
            expected_offsets[ids] = torch.arange(size, device=ids.device)
            if tuple(self.token_codes[group].shape) != (size, self.code_dims[group]):
                raise ValueError(f"Tiered RankLift token-code shape mismatch in group {group}")
            if tuple(self.right_factors[group].shape) != (
                self.embed_dim, self.feature_dims[group]
            ):
                raise ValueError(f"Tiered RankLift right-factor shape mismatch in group {group}")
            start += size
        if not torch.equal(self.group_offsets, expected_offsets):
            raise ValueError("Tiered RankLift token offsets are inconsistent")

    def token_ids_for_group(self, group: int) -> torch.Tensor:
        if not 0 <= group < self.num_groups:
            raise IndexError(f"group index out of range: {group}")
        start = sum(self.group_sizes[:group])
        return self.token_ids_by_group[start:start + self.group_sizes[group]]

    def features_from_codes(
        self, group: int, codes: torch.Tensor
    ) -> torch.Tensor:
        if not 0 <= group < self.num_groups:
            raise IndexError(f"group index out of range: {group}")
        if codes.shape[-1] != self.code_dims[group]:
            raise ValueError(
                f"group {group} codes must end in width {self.code_dims[group]}, "
                f"got {codes.shape[-1]}"
            )
        if self.lift_dims[group] == 0:
            return codes
        normalized = _parameter_free_rms_norm(codes, self.rms_eps)
        lifted = (
            F.silu(self.lift_a[group](normalized))
            * self.lift_b[group](normalized)
        )
        return torch.cat((codes, lifted), dim=-1)

    def materialize_group_features(self, group: int) -> torch.Tensor:
        return self.features_from_codes(group, self.token_codes[group])

    def materialize_effective_table(self) -> torch.Tensor:
        grouped = torch.cat([
            self.materialize_group_features(group)
            @ self.right_factors[group].T
            for group in range(self.num_groups)
        ])
        return grouped.index_select(0, self.inverse_grouped_order)

    def forward(self, input_ids, doc_mask=None):
        flat_ids = input_ids.reshape(-1)
        flat_groups = self.group_ids[flat_ids]
        flat_offsets = self.group_offsets[flat_ids]
        output = torch.empty(
            flat_ids.numel(), self.embed_dim,
            device=flat_ids.device,
            dtype=self.token_codes[0].dtype,
        )
        # Execute every group even when the current input contains no member of
        # it.  Empty tensor operations keep all parameters connected under DDP
        # with find_unused_parameters=False, matching GroupReduce's behavior.
        for group in range(self.num_groups):
            mask = flat_groups == group
            codes = self.token_codes[group][flat_offsets[mask]]
            features = self.features_from_codes(group, codes)
            output[mask] = features @ self.right_factors[group].T
        return output.view(*input_ids.shape, self.embed_dim), None


class FunnelingEmbed(nn.Module):
    """Width-preserving nonlinear factorization control.

    This is the architectural part of Lioutas et al.'s Distilled Embedding:
    ``E = ReLU(Z) P^T``.  It intentionally omits their dense-teacher
    reconstruction and distillation curriculum, so a run from random
    initialization must be reported as a from-scratch control, not a faithful
    reproduction of the full published procedure.
    """

    def __init__(self, vocab_size: int, embed_dim: int, rank: int = 128) -> None:
        super().__init__()
        _validate_common(vocab_size, embed_dim, rank)
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.rank = int(rank)

        self.token_codes = nn.Parameter(
            torch.randn(self.vocab_size, self.rank) * 0.02
        )
        self.projection = nn.Linear(self.rank, self.embed_dim, bias=True)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def materialize_features(self) -> torch.Tensor:
        return F.relu(self.token_codes)

    def materialize_effective_table(self) -> torch.Tensor:
        return self.projection(self.materialize_features())

    def forward(self, input_ids, doc_mask=None):
        features = F.relu(F.embedding(input_ids, self.token_codes))
        return self.projection(features), None


class DeFINEEmbed(nn.Module):
    """DeFINE map-expand-reduce with a tied low-dimensional classifier.

    The implementation follows the paper's hierarchical group transform and
    direct map-layer mixer:

    * layer 0 splits the map vector into groups and applies one independent
      linear transform to every group;
    * later layers split both the previous representation and original map
      vector into the current number of groups, concatenate corresponding
      chunks, and apply independent transforms to the groups;
    * a final dense reduce projection maps the maximum expansion width to the
      Qwen hidden dimension.

    The paper's parameter formula is ``input_dim * output_dim / groups``;
    therefore the matrices cannot be shared across groups (which would divide
    by ``groups`` a second time).  The authors' released implementation
    confirms a weight tensor of shape ``(groups, input/group, output/group)``
    and applies bias, group-wise LayerNorm, then GELU.  We retain those
    operations but set dropout to zero to match this repository's Qwen
    pretraining control.  The flat output classifier first maps hidden states
    back to ``code_dim`` and then multiplies by the same ``token_codes`` table.
    Thus the low-dimensional map is tied, while the expanded input table is
    not claimed to be exactly tied.

    This is a fixed-tokenizer, flat-softmax Qwen adaptation.  The published
    Transformer-XL experiments combined DeFINE with projective/adaptive
    input-output machinery and therefore are not protocol-identical.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        code_dim: int = 112,
        expansion_dims: Sequence[int] = (656, 1184, 1724),
        group_counts: Sequence[int] = (16, 8, 4),
    ) -> None:
        super().__init__()
        _validate_common(vocab_size, embed_dim, code_dim)
        expansion_dims = tuple(int(value) for value in expansion_dims)
        group_counts = tuple(int(value) for value in group_counts)
        if not expansion_dims or len(expansion_dims) != len(group_counts):
            raise ValueError(
                "expansion_dims and group_counts must be non-empty and equal-length"
            )
        if any(value <= 0 for value in expansion_dims + group_counts):
            raise ValueError("DeFINE dimensions and group counts must be positive")
        if any(
            group_counts[index + 1] > group_counts[index]
            for index in range(len(group_counts) - 1)
        ):
            raise ValueError("DeFINE group counts must be non-increasing")

        previous_dim = code_dim
        for index, (output_dim, groups) in enumerate(
            zip(expansion_dims, group_counts)
        ):
            input_dim = code_dim if index == 0 else previous_dim + code_dim
            if input_dim % groups != 0 or output_dim % groups != 0:
                raise ValueError(
                    f"DeFINE layer {index} dimensions ({input_dim}->{output_dim}) "
                    f"must be divisible by groups={groups}"
                )
            previous_dim = output_dim

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.code_dim = int(code_dim)
        self.expansion_dims = expansion_dims
        self.group_counts = group_counts

        self.token_codes = nn.Parameter(
            torch.randn(self.vocab_size, self.code_dim) * 0.02
        )
        weights = []
        biases = []
        norms = []
        previous_dim = self.code_dim
        for index, (output_dim, groups) in enumerate(
            zip(self.expansion_dims, self.group_counts)
        ):
            input_dim = (
                self.code_dim if index == 0
                else previous_dim + self.code_dim
            )
            # DeFINE/GroupLinear owns an independent matrix and bias for every
            # group.  Sharing this matrix changes the paper's parameter count
            # from input_dim*output_dim/groups to
            # input_dim*output_dim/groups**2 and is not HGT.
            weight = nn.Parameter(torch.empty(
                groups, input_dim // groups, output_dim // groups
            ))
            bias = nn.Parameter(torch.zeros(
                groups, output_dim // groups
            ))
            nn.init.xavier_uniform_(weight)
            weights.append(weight)
            biases.append(bias)
            norms.append(nn.LayerNorm(output_dim // groups))
            previous_dim = output_dim
        self.expand_weights = nn.ParameterList(weights)
        self.expand_biases = nn.ParameterList(biases)
        self.expand_norms = nn.ModuleList(norms)
        self.reduce = nn.Linear(
            self.expansion_dims[-1], self.embed_dim, bias=True
        )
        self.output_projection = nn.Linear(
            self.embed_dim, self.code_dim, bias=False
        )
        nn.init.xavier_uniform_(self.reduce.weight)
        nn.init.zeros_(self.reduce.bias)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)

    def expand_codes(self, codes: torch.Tensor) -> torch.Tensor:
        original = codes
        state = codes
        for index, (output_dim, groups, weight, bias, norm) in enumerate(zip(
            self.expansion_dims,
            self.group_counts,
            self.expand_weights,
            self.expand_biases,
            self.expand_norms,
        )):
            if index == 0:
                mixed = original.reshape(
                    *original.shape[:-1], groups, self.code_dim // groups
                )
            else:
                previous_chunks = state.reshape(
                    *state.shape[:-1], groups, state.shape[-1] // groups
                )
                original_chunks = original.reshape(
                    *original.shape[:-1], groups, self.code_dim // groups
                )
                mixed = torch.cat((previous_chunks, original_chunks), dim=-1)
            # Authors' GroupLinear path: flatten leading dimensions, move the
            # group axis first, and issue one batched GEMM.
            flat_mixed = mixed.reshape(-1, groups, mixed.shape[-1])
            state = torch.bmm(flat_mixed.transpose(0, 1), weight)
            state = state.transpose(0, 1).reshape(
                *codes.shape[:-1], groups, output_dim // groups
            )
            state = F.gelu(norm(state + bias)).reshape(
                *codes.shape[:-1], output_dim
            )
        return state

    def materialize_input_table(self) -> torch.Tensor:
        return self.reduce(self.expand_codes(self.token_codes))

    def forward(self, input_ids, doc_mask=None):
        expanded = self.expand_codes(F.embedding(input_ids, self.token_codes))
        return self.reduce(expanded), None
