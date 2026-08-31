#!/usr/bin/env python3
"""Standalone CUDA-autocast gate for residual subspace experts.

This script intentionally depends only on PyTorch and project modules so it
can run inside the minimal production training environment (which need not
install pytest).
"""

import sys
from pathlib import Path

# Direct execution sets sys.path[0] to scripts/, not the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from compositional.residual_subspace_experts import (
    ResidualSubspaceExpertsEmbed,
)
from compositional.tied_head import make_tied_head


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the RSE autocast gate")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA support is required for the RSE gate")

    torch.manual_seed(13)
    device = torch.device("cuda:0")
    embed = ResidualSubspaceExpertsEmbed(
        19,
        11,
        base_rank=5,
        expert_rank=4,
        num_experts=4,
        router_dim=3,
        top_k=2,
        router_temperature=0.75,
    ).to(device)
    with torch.no_grad():
        embed.expert_down_weight.normal_(std=0.1)
        embed.expert_down_bias.normal_(std=0.1)
        embed.expert_up_bias.normal_(std=0.1)

    ids = torch.tensor([[0, 3, 7, 18]], device=device)
    hidden = torch.randn(2, 3, 11, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embeddings, theta = embed(ids)
        if theta.dtype != torch.float32:
            raise AssertionError(f"expected FP32 routing, got {theta.dtype}")
        logits = make_tied_head(
            embed, "residual_subspace_experts", 19
        )(hidden)
        auxiliary = embed.pop_router_aux_loss()
        if auxiliary is None:
            raise AssertionError("router auxiliary loss was not produced")
        reference = hidden @ embed.materialize().T
        loss = logits.float().square().mean() + 0.01 * auxiliary.float()

    if embeddings.dtype != torch.bfloat16:
        raise AssertionError(f"unexpected embedding dtype {embeddings.dtype}")
    if logits.dtype != torch.bfloat16:
        raise AssertionError(f"unexpected logit dtype {logits.dtype}")
    for name, tensor in (
        ("embeddings", embeddings),
        ("logits", logits),
        ("reference", reference),
        ("auxiliary", auxiliary),
        ("loss", loss),
    ):
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"non-finite {name}")
    torch.testing.assert_close(logits, reference, rtol=2e-2, atol=2e-2)

    loss.backward()
    for name, parameter in embed.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"missing AMP gradient for {name}")
        if not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"non-finite AMP gradient for {name}")

    torch.cuda.synchronize(device)
    print("RSE_CUDA_BF16_AUTOCAST_STANDALONE_PASS", flush=True)


if __name__ == "__main__":
    main()
