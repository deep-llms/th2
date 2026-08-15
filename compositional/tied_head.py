"""Tied output head: replaces lm_head to share weights with the input embedding.

Instead of a free V×d output matrix (155.6M params for Qwen3), the output
logits are computed from the same weights as the input embedding:

    logits = hidden @ embed_table(all_tokens).T

This is the standard weight-tying approach (Press & Wolf 2017, ALBERT).
For each embedding architecture, we use the most efficient computation:

  LowRankEmbed:   logits = (hidden @ proj.weight) @ X.T     (factored, no materialization)
  OriginalANT:    logits = (hidden @ A.T) @ T.T              (factored via T and A)
  ANTEmbed:       logits = hidden @ materialize(all_ids).T    (must materialize, batched)
  ResidualANT:    same as ANT (materializes identity + codebook)

V2Embed (context-dependent routing) cannot be tied — the embedding per token
depends on surrounding context, so there is no fixed per-token output vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TiedLowRankHead(nn.Module):
    """Efficient tied output for LowRankEmbed: factored H→E→V projection."""

    def __init__(self, embed):
        super().__init__()
        self.embed = embed

    def forward(self, hidden_states):
        # proj is Linear(E→H), weight shape (H, E)
        # Forward: x @ weight.T = (B,E) @ (E,H) = (B,H)
        # Reverse: hidden @ weight = (B,H) @ (H,E) = (B,E)
        h_proj = hidden_states @ self.embed.proj.weight  # (B, L, E)
        return h_proj @ self.embed.X.T  # (B, L, V)


class TiedOriginalANTHead(nn.Module):
    """Efficient tied output for OriginalANT: factored via T and A."""

    def __init__(self, embed):
        super().__init__()
        self.embed = embed

    def forward(self, hidden_states):
        # E_eff = T @ A, so logits = hidden @ (T @ A).T = hidden @ A.T @ T.T
        # hidden: (B, L, d), A: (K, d), T: (V, K)
        h_code = hidden_states @ self.embed.A.T  # (B, L, K)
        return h_code @ self.embed.T.T  # (B, L, V)


class TiedMaterializeHead(nn.Module):
    """Tied output by materializing the full embedding table.

    Used for ANTEmbed and ResidualANTEmbed where the per-token embedding
    is a nonlinear function of the token id (entmax routing). Computes
    embed(all_ids) in batches and uses the result as the output projection.

    Cost: one extra forward through the embedding module per training step
    (~37 batches of 4096 for V=151936). The embedding module is small
    (~24M params), so this adds ~10-20ms per step on H100.
    """

    def __init__(self, embed, vocab_size, batch_size=4096):
        super().__init__()
        self.embed = embed
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
        embed_type: one of "lowrank", "original_ant", "ant", "residual_ant"
        vocab_size: vocabulary size

    Returns:
        nn.Module that computes logits from hidden states using tied weights
    """
    if embed_type == "lowrank":
        return TiedLowRankHead(embed)
    if embed_type == "original_ant":
        return TiedOriginalANTHead(embed)
    if embed_type in ("ant", "residual_ant"):
        return TiedMaterializeHead(embed, vocab_size)
    raise ValueError(f"Cannot tie output for embed_type={embed_type} "
                     "(v2/context-dependent routing cannot be tied)")
