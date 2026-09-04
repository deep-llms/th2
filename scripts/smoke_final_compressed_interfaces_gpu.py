#!/usr/bin/env python3
"""Production-shape CUDA/DDP smoke for the final tied-interface candidates.

The smoke constructs the real Qwen3-0.6B vocabulary and hidden width for the
selected compressed interface.  Each visible GPU owns one DDP rank.
For every arm it runs the input path, tied output path, backward, one AdamW
update, and a NCCL checksum.  Rank 0 writes a JSON report when ``--output`` is
provided.

This is deliberately an interface smoke rather than a language-model training
benchmark: it isolates the embedding/classifier implementation while retaining
the production parameter shapes and BF16 arithmetic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compositional.compressed_baselines import (
    GroupReduceEmbed,
    SlimEmbed,
    TTEmbedding,
)
from compositional.compression_init import (
    frequency_group_ids_from_populations,
    load_frequency_counts,
)
from compositional.nonlinear_factorizations import (
    DeFINEEmbed,
    FunnelingEmbed,
    RankLiftEmbed,
    TieredRankLiftEmbed,
)
from compositional.tied_head import make_tied_head


VOCAB_SIZE = 151_936
HIDDEN_SIZE = 1_024
TIER_POPULATIONS = (2_048, 6_144, 24_576, 119_168)


def _language_balanced_groups() -> torch.Tensor:
    importance_path = (
        PROJECT_ROOT / "resources" / "token_importance_langbalanced.npz"
    )
    importance = load_frequency_counts(
        importance_path, VOCAB_SIZE, key="counts", pseudocount=0.0
    )
    return frequency_group_ids_from_populations(
        importance, TIER_POPULATIONS
    )


def _build(name: str) -> nn.Module:
    if name == "groupreduce_matched_lb_t4":
        return GroupReduceEmbed(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            group_ranks=(1_024, 512, 192, 64),
            group_ids=_language_balanced_groups(),
        )
    if name == "tiered_ranklift_lb_t4_c512":
        return TieredRankLiftEmbed(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            code_dims=(1_024, 512, 192, 64),
            lift_dims=(0, 0, 320, 192),
            group_ids=_language_balanced_groups(),
            rms_eps=1e-6,
        )
    if name == "ranklift_tied_c124_m460":
        return RankLiftEmbed(
            VOCAB_SIZE, HIDDEN_SIZE, code_dim=124, lift_dim=336,
            rms_eps=1e-6,
        )
    if name == "funneling_tied_r128":
        return FunnelingEmbed(VOCAB_SIZE, HIDDEN_SIZE, rank=128)
    if name == "define_tied_n112_k1724":
        return DeFINEEmbed(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            code_dim=112,
            expansion_dims=(656, 1184, 1724),
            group_counts=(16, 8, 4),
        )
    if name == "slim_tied_k4_m76484":
        return SlimEmbed(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            num_components=4,
            num_subvectors=76_484,
            mapping_seed=42,
        )
    if name == "tt_tied_r219":
        return TTEmbedding(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            vocab_modes=(50, 50, 64),
            embedding_modes=(8, 8, 16),
            tt_ranks=(1, 219, 219, 1),
            implementation="materialize",
            materialize_chunk_size=1024,
        )
    raise ValueError(f"unknown smoke arm: {name}")


def _arm_type(name: str) -> str:
    return {
        "groupreduce_matched_lb_t4": "groupreduce",
        "tiered_ranklift_lb_t4_c512": "tiered_ranklift",
        "ranklift_tied_c124_m460": "ranklift",
        "funneling_tied_r128": "funneling",
        "define_tied_n112_k1724": "define",
        "slim_tied_k4_m76484": "slim",
        "tt_tied_r219": "tt",
    }[name]


class _InterfaceStep(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.embed = _build(name)
        # The head deliberately keeps a non-registering reference.  DDP must
        # see each shared parameter exactly once, under ``embed``.
        self.head = make_tied_head(self.embed, _arm_type(name), VOCAB_SIZE)

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor):
        embeddings, auxiliary = self.embed(input_ids)
        if auxiliary is not None:
            raise RuntimeError("final interface smoke expected no auxiliary loss")
        logits = self.head(hidden)
        # Touch both input and output paths.  Selecting a few logit columns
        # keeps the scalar reduction cheap without changing head computation.
        probe = logits[..., (0, 1, 17, VOCAB_SIZE - 1)]
        return embeddings.float().square().mean() + probe.float().square().mean()


def _run_rank(rank: int, world: int, args) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.manual_seed(20260902)
    torch.cuda.manual_seed_all(20260902)

    names = args.arms.split(",")
    reports = []
    for name in names:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        module = _InterfaceStep(name).to(device=device, dtype=torch.bfloat16)
        ddp = DDP(module, device_ids=[rank], find_unused_parameters=False)
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=3e-4)
        generator = torch.Generator(device=device).manual_seed(9000 + rank)
        input_ids = torch.randint(
            0,
            VOCAB_SIZE,
            (args.batch_size, args.sequence_length),
            device=device,
            generator=generator,
        )
        hidden = torch.randn(
            args.batch_size,
            args.sequence_length,
            HIDDEN_SIZE,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )

        before = next(ddp.parameters()).detach().clone()
        dist.barrier()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ddp(input_ids, hidden)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name}: non-finite loss on rank {rank}: {loss}")
        loss.backward()
        missing = [
            parameter_name
            for parameter_name, parameter in ddp.module.embed.named_parameters()
            if parameter.grad is None
        ]
        nonfinite = [
            parameter_name
            for parameter_name, parameter in ddp.module.embed.named_parameters()
            if parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
        ]
        if missing or nonfinite:
            raise RuntimeError(
                f"{name}: bad gradients missing={missing}, nonfinite={nonfinite}"
            )
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        changed = not torch.equal(before, next(ddp.parameters()).detach())
        if not changed:
            raise RuntimeError(f"{name}: optimizer did not update first parameter")

        marker = torch.tensor(
            [float(loss.detach()), float(rank + 1)], device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(marker, op=dist.ReduceOp.SUM)
        expected_rank_sum = world * (world + 1) / 2
        if marker[1].item() != expected_rank_sum:
            raise RuntimeError(
                f"{name}: NCCL marker {marker[1].item()} != {expected_rank_sum}"
            )
        peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
        local = {
            "arm": name,
            "rank": rank,
            "loss": float(loss.detach()),
            "step_seconds": elapsed,
            "peak_allocated_gib": peak_gib,
            "parameters": sum(p.numel() for p in ddp.module.embed.parameters()),
            "nccl_rank_sum": marker[1].item(),
        }
        gathered = [None] * world if rank == 0 else None
        dist.gather_object(local, gathered, dst=0)
        if rank == 0:
            reports.append({
                "arm": name,
                "parameters": local["parameters"],
                "world_size": world,
                "mean_step_seconds": sum(x["step_seconds"] for x in gathered) / world,
                "max_peak_allocated_gib": max(x["peak_allocated_gib"] for x in gathered),
                "losses": [x["loss"] for x in gathered],
                "nccl_rank_sum": local["nccl_rank_sum"],
                "status": "PASS",
            })
            print(json.dumps(reports[-1], sort_keys=True), flush=True)
        del optimizer, ddp, module, input_ids, hidden, before, loss
        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0:
        result = {
            "status": "PASS",
            "world_size": world,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "batch_size_per_rank": args.batch_size,
            "sequence_length": args.sequence_length,
            "arms": reports,
        }
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n")
        print("FINAL_COMPRESSED_INTERFACES_GPU_SMOKE_PASS", flush=True)
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arms",
        default=(
            "ranklift_tied_c124_m460,funneling_tied_r128,"
            "define_tied_n112_k1724,slim_tied_k4_m76484,tt_tied_r219"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.sequence_length <= 0:
        raise SystemExit("batch size and sequence length must be positive")
    names = args.arms.split(",")
    if not names or any(not name for name in names):
        raise SystemExit("--arms must be a non-empty comma-separated list")
    for name in names:
        _arm_type(name)

    world = torch.cuda.device_count()
    if world <= 0:
        raise SystemExit("no visible CUDA devices")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29529")
    if world == 1:
        _run_rank(0, 1, args)
    else:
        mp.spawn(_run_rank, args=(world, args), nprocs=world, join=True)


if __name__ == "__main__":
    main()
