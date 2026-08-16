"""Tied output head: replaces lm_head to share weights with the input embedding.

Instead of a free V×d output matrix (155.6M params for Qwen3), the output
logits are computed from the same weights as the input embedding:

    logits = hidden @ embed_table(all_tokens).T

This is the standard weight-tying approach (Press & Wolf 2017, ALBERT).
For each embedding architecture, we use the most efficient computation:

  LowRankEmbed:   logits = (hidden @ proj.weight) @ X.T     (factored, no materialization)
  SharedLocal:    one grouped dense projection over concatenated shared/local factors
  OriginalANT:    logits = (hidden @ A.T) @ T.T              (factored via T and A)
  ANTEmbed:       logits = hidden @ materialize(all_ids).T    (must materialize, batched)
  ResidualANT:    same as ANT (materializes identity + codebook)

V0Embed, V1Embed, V2Embed, and IsolationControlEmbed cannot be tied because
their embedding for a token depends on surrounding context, so there is no
fixed per-token output vector.
"""

import torch
import torch.nn as nn


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
        h_proj = hidden_states @ self.embed.proj.weight  # (B, L, E)
        logits = h_proj @ self.embed.X.T  # (B, L, V)
        if self.embed.proj.bias is not None:
            # bias (H,): each hidden position dot-products with bias,
            # adds a scalar to all V logits (shifts all equally)
            logits = logits + (hidden_states @ self.embed.proj.bias).unsqueeze(-1)
        return logits


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
        embed_type: one of "lowrank", "shared_local", "original_ant", "ant",
                    or "residual_ant"
        vocab_size: vocabulary size

    Returns:
        nn.Module that computes logits from hidden states using tied weights
    """
    if embed_type == "lowrank":
        return TiedLowRankHead(embed)
    if embed_type == "shared_local":
        return TiedSharedLocalHead(embed)
    if embed_type == "original_ant":
        return TiedOriginalANTHead(embed)
    if embed_type in ("ant", "residual_ant"):
        return TiedMaterializeHead(embed, vocab_size)
    raise ValueError(
        f"Cannot tie output for embed_type={embed_type}. Supported types are "
        "lowrank, shared_local, original_ant, ant, and residual_ant; v0, v1, "
        "v2, and isolation_control are context-dependent and cannot be tied."
    )
