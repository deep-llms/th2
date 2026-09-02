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
