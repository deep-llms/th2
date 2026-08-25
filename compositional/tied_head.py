"""Compressed output heads for compositional embedding experiments.

Tied heads replace the free V×d output matrix (155.6M params for Qwen3) and
compute logits from the same weights as the input embedding:

    logits = hidden @ embed_table(all_tokens).T

This is the standard weight-tying approach (Press & Wolf 2017, ALBERT).
For each embedding architecture, we use the most efficient computation:

  LowRankEmbed:   logits = (hidden @ proj.weight) @ X.T     (factored, no materialization)
  SharedLocal:    one grouped dense projection over concatenated shared/local factors
  PureLocal:      one grouped dense projection through independent local bases
  P-VQ:           codebook projection + code lookup + exclusive projection
  Slim:           component GEMMs + fixed mapping-table gather/sum
  GroupReduce:    one factored projection per vocabulary block
  TT:             reverse TT-matrix contraction
  OriginalANT:    logits = (hidden @ A.T) @ T.T              (factored via T and A)
  ANTEmbed:       logits = hidden @ materialize(all_ids).T    (must materialize, batched)
  ResidualANT:    same as ANT (materializes identity + codebook)

V0Embed, V1Embed, V2Embed, and IsolationControlEmbed cannot be tied because
their embedding for a token depends on surrounding context, so there is no
fixed per-token output vector.

``IndependentLowRankHead`` is a causal diagnostic rather than a tied head. It
starts with value-identical copies of a ``LowRankEmbed`` token table and
projection weight, but owns separate Parameters. This holds the input
architecture and output rank fixed while isolating hard parameter sharing.
"""

import torch
import torch.nn as nn


INDEPENDENT_OUTPUT_FILENAME = "output_head.pt"


def _factorized_low_rank_logits(hidden_states, token_factors, proj_weight,
                                projection_bias=None):
    """Compute exact logits for a V×r factorization without materializing V×d."""
    h_proj = hidden_states @ proj_weight
    logits = h_proj @ token_factors.T
    if projection_bias is not None:
        # A rank->hidden embedding bias contributes h·b equally to every class.
        # It is included only for exact tying to the input embedding table.
        logits = logits + (hidden_states @ projection_bias).unsqueeze(-1)
    return logits


class _TiedHeadBase(nn.Module):
    """Base class that keeps a non-registering reference to the input embedder.

    The embedding module is already registered under ``model.embed_tokens``.
    Registering it again under ``lm_head`` makes the state dict contain two
    names for every shared tensor, which Hugging Face safe serialization
    rejects.  Bypassing ``nn.Module.__setattr__`` here preserves a strong
    reference without creating a second module/parameter owner.
    """

    def __init__(self, embed):
        super().__init__()
        object.__setattr__(self, "_embed_ref", embed)

    @property
    def embed(self):
        return self._embed_ref


class TiedLowRankHead(_TiedHeadBase):
    """Efficient tied output for LowRankEmbed: factored H→E→V projection."""

    def __init__(self, embed):
        super().__init__(embed)

    def forward(self, hidden_states):
        # proj is Linear(E→H), weight shape (H, E)
        # Input: e = X[i] @ W.T + b  (per token, includes bias)
        # Tied output: logits = hidden @ table.T where table = X @ W.T + b
        # = hidden @ (X @ W.T + b).T
        # = hidden @ (W @ X.T) + hidden @ b (b broadcast across V)
        # Factored: H→E then E→V, plus bias contribution
        return _factorized_low_rank_logits(
            hidden_states,
            self.embed.X,
            self.embed.proj.weight,
            self.embed.proj.bias,
        )


class IndependentLowRankHead(nn.Module):
    """Independent rank-r output initialized from a ``LowRankEmbed``.

    Only the two classifier-relevant factors are copied. ``LowRankEmbed``'s
    rank->hidden bias is omitted because it would add the same scalar to every
    vocabulary logit and therefore cancels exactly in softmax. A hidden->rank
    bias or a V-way class bias would instead change the architecture and is not
    part of Qwen's bias-free ``lm_head``.

    Construction performs clones only—no random initialization—so adding this
    control does not advance PyTorch's RNG relative to the tied run.
    """

    def __init__(self, input_embed):
        super().__init__()
        if not hasattr(input_embed, "X") or not hasattr(input_embed, "proj"):
            raise TypeError(
                "IndependentLowRankHead requires a LowRankEmbed-like module"
            )
        if input_embed.proj.weight.ndim != 2 or input_embed.X.ndim != 2:
            raise ValueError("Low-rank factors must both be matrices")
        if input_embed.X.shape[1] != input_embed.proj.weight.shape[1]:
            raise ValueError(
                "Token-factor rank does not match projection-weight rank"
            )

        self.X = nn.Parameter(input_embed.X.detach().clone())
        self.proj_weight = nn.Parameter(
            input_embed.proj.weight.detach().clone()
        )

    @property
    def rank(self):
        return self.X.shape[1]

    def forward(self, hidden_states):
        return _factorized_low_rank_logits(
            hidden_states, self.X, self.proj_weight
        )


class TiedSharedLocalHead(_TiedHeadBase):
    """Exact tied output for SharedLocalEmbed using standard dense GEMMs."""

    def __init__(self, embed):
        super().__init__(embed)

    def forward(self, hidden_states):
        # Project once into the shared subspace and once into every group-local
        # subspace.  The token coefficients are already stored with matching
        # shared/local channels, so concatenating only the small hidden-side
        # latents avoids constructing and adding two full-V logit tensors. The
        # common divisible-vocabulary path ends in one ordinary batched GEMM.
        flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
        grouped_hidden = flat_hidden.unsqueeze(0).expand(
            self.embed.num_groups, -1, -1
        )
        h_shared = flat_hidden @ self.embed.shared_proj.weight
        h_local = torch.bmm(grouped_hidden, self.embed.local_weight)
        grouped_latent = torch.cat((
            h_shared.unsqueeze(0).expand(self.embed.num_groups, -1, -1),
            h_local,
        ), dim=-1)
        if self.embed.num_large_groups == 0:
            # Production fast path: Qwen's vocabulary divides evenly, so the
            # group-major layout is already exact token-id order with no gaps.
            grouped_logits = torch.bmm(
                grouped_latent, self.embed.token_factors.transpose(1, 2)
            )
            logits = grouped_logits.permute(1, 0, 2).reshape(
                flat_hidden.size(0), -1
            )
        else:
            # Variable-size groups cannot be represented by one strided bmm
            # without padding between token-id ranges. A few ordinary GEMMs
            # avoid computing/scattering those padded outputs; concatenation in
            # group order is exact token-id order.
            logits = torch.cat([
                grouped_latent[group] @ self.embed.token_factors[
                    group, :self.embed.group_sizes[group],
                ].T
                for group in range(self.embed.num_groups)
            ], dim=-1)
        logits = logits.view(*hidden_states.shape[:-1], -1)

        if self.embed.shared_proj.bias is not None:
            logits = logits + (
                hidden_states @ self.embed.shared_proj.bias
            ).unsqueeze(-1)
        return logits


class TiedPureLocalHead(_TiedHeadBase):
    """Exact tied output for PureLocalEmbed using standard grouped GEMMs."""

    def forward(self, hidden_states):
        flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
        grouped_hidden = flat_hidden.unsqueeze(0).expand(
            self.embed.num_groups, -1, -1
        )
        grouped_latent = torch.bmm(grouped_hidden, self.embed.local_weight)

        if self.embed.num_large_groups == 0:
            grouped_logits = torch.bmm(
                grouped_latent, self.embed.token_factors.transpose(1, 2)
            )
            logits = grouped_logits.permute(1, 0, 2).reshape(
                flat_hidden.size(0), -1
            )
        else:
            logits = torch.cat([
                grouped_latent[group] @ self.embed.token_factors[
                    group, :self.embed.group_sizes[group],
                ].T
                for group in range(self.embed.num_groups)
            ], dim=-1)

        logits = logits.view(*hidden_states.shape[:-1], -1)
        # The one input-side bias belongs to every effective token embedding,
        # so exact tying contributes the same h·b scalar to every class.
        return logits + (hidden_states @ self.embed.bias).unsqueeze(-1)


class TiedPVQHead(_TiedHeadBase):
    """Exact compact P-VQ output without reconstructing the V x d table."""

    def forward(self, hidden_states):
        shared_hidden = hidden_states[..., :self.embed.shared_dim]
        exclusive_hidden = hidden_states[..., self.embed.shared_dim:]

        # Compute one score per shared code, then expand to token order with
        # the fixed assignment.  The exclusive slice preserves a unique
        # learned contribution for every token.
        shared_code_logits = shared_hidden @ self.embed.codebook.T
        shared_token_logits = shared_code_logits.index_select(
            -1, self.embed.assignments
        )
        exclusive_logits = exclusive_hidden @ self.embed.exclusive.T
        return shared_token_logits + exclusive_logits


class TiedSlimHead(_TiedHeadBase):
    """Exact Slim output: component GEMMs followed by indexed summation."""

    def forward(self, hidden_states):
        flat_hidden = hidden_states.reshape(-1, self.embed.embed_dim)
        component_hidden = flat_hidden.reshape(
            flat_hidden.size(0),
            self.embed.num_components,
            self.embed.component_dim,
        ).permute(1, 0, 2)
        partial = torch.bmm(
            component_hidden, self.embed.subvectors.transpose(1, 2)
        )

        # Each iteration materializes only one N x V contribution.  Stacking
        # every gathered component at once would inflate activation memory by
        # num_components while producing exactly the same result.
        logits = partial[0].index_select(-1, self.embed.mapping[:, 0])
        for component in range(1, self.embed.num_components):
            logits = logits + partial[component].index_select(
                -1, self.embed.mapping[:, component]
            )
        return logits.view(*hidden_states.shape[:-1], self.embed.vocab_size)


class TiedGroupReduceHead(_TiedHeadBase):
    """Exact tied block-factor output for a GroupReduce representation."""

    def forward(self, hidden_states):
        flat_hidden = hidden_states.reshape(-1, self.embed.embed_dim)
        grouped_logits = []
        for group in range(self.embed.num_groups):
            projected = flat_hidden @ self.embed.right_factors[group]
            grouped_logits.append(
                projected @ self.embed.left_factors[group].T
            )
        logits = torch.cat(grouped_logits, dim=-1).index_select(
            -1, self.embed.inverse_grouped_order
        )
        return logits.view(*hidden_states.shape[:-1], self.embed.vocab_size)


class TiedTTHead(_TiedHeadBase):
    """Exact tied output using the TT-matrix contraction implemented by TTEmbedding."""

    def forward(self, hidden_states):
        return self.embed.project_hidden(hidden_states)


class TiedOriginalANTHead(_TiedHeadBase):
    """Efficient tied output for OriginalANT: factored via T and A."""

    def __init__(self, embed):
        super().__init__(embed)

    def forward(self, hidden_states):
        # E_eff = T @ A, so logits = hidden @ (T @ A).T = hidden @ A.T @ T.T
        # hidden: (B, L, d), A: (K, d), T: (V, K)
        h_code = hidden_states @ self.embed.A.T  # (B, L, K)
        return h_code @ self.embed.T.T  # (B, L, V)


class TiedMaterializeHead(_TiedHeadBase):
    """Tied output by materializing the full embedding table.

    Used for ANTEmbed and ResidualANTEmbed where the per-token embedding
    is a nonlinear function of the token id (entmax routing). Computes
    embed(all_ids) in batches and uses the result as the output projection.

    Cost: one extra full-vocabulary forward through the embedding module per
    micro-batch (~37 chunks of 4096 for V=151936). Benchmark this overhead on
    the target hardware when choosing an accumulation schedule.
    """

    def __init__(self, embed, vocab_size, batch_size=4096):
        super().__init__(embed)
        self.vocab_size = vocab_size
        self.batch_size = batch_size

    def forward(self, hidden_states):
        device = hidden_states.device
        all_ids = torch.arange(self.vocab_size, device=device)

        embs = []
        for start in range(0, self.vocab_size, self.batch_size):
            ids = all_ids[start:start + self.batch_size].unsqueeze(0)
            e, _ = self.embed(ids)
            embs.append(e.squeeze(0))
        embed_table = torch.cat(embs, dim=0)  # (V, d)

        return hidden_states @ embed_table.T  # (B, L, V)


def make_tied_head(embed, embed_type, vocab_size):
    """Create the appropriate tied head for an embedding module.

    Args:
        embed: the compositional embedding module
        embed_type: one of the supported context-independent embedding arms
        vocab_size: vocabulary size

    Returns:
        nn.Module that computes logits from hidden states using tied weights
    """
    if embed_type in ("lowrank", "global_lowrank"):
        return TiedLowRankHead(embed)
    if embed_type == "shared_local":
        return TiedSharedLocalHead(embed)
    if embed_type == "pure_local":
        return TiedPureLocalHead(embed)
    if embed_type == "pvq":
        return TiedPVQHead(embed)
    if embed_type == "slim":
        return TiedSlimHead(embed)
    if embed_type == "groupreduce":
        return TiedGroupReduceHead(embed)
    if embed_type == "tt":
        return TiedTTHead(embed)
    if embed_type == "original_ant":
        return TiedOriginalANTHead(embed)
    if embed_type in ("ant", "residual_ant"):
        return TiedMaterializeHead(embed, vocab_size)
    raise ValueError(
        f"Cannot tie output for embed_type={embed_type}. Supported types are "
        "lowrank/global_lowrank, shared_local, pure_local, pvq, slim, "
        "groupreduce, tt, "
        "original_ant, ant, and residual_ant; v0, v1, v2, and "
        "isolation_control are context-dependent and cannot be tied."
    )
