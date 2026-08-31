#!/usr/bin/env python3
"""CUDA and NCCL/DDP smoke test for residual subspace experts.

This is intentionally a synthetic test: it uses the production embedding
dimensions without loading Qwen or the training dataset.  It checks the custom
embedding and tied output head themselves, including BF16 forward/backward,
optimizer updates, finite gradients, numerical tying, and a real DDP gradient
reduction with ``find_unused_parameters=False``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compositional.residual_subspace_experts import (  # noqa: E402
    ResidualSubspaceExpertsEmbed,
)
from compositional.tied_head import make_tied_head  # noqa: E402


VOCAB_SIZE = 151_936
EMBED_DIM = 1_024
BASE_RANK = 120
EXPERT_RANK = 80
NUM_EXPERTS = 12
ROUTER_DIM = 32
TOP_K = 2
ROUTER_TEMPERATURE = 1.0


class Tee:
    """Write parent-process output to both the terminal and a log file."""

    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()


class SmokeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = ResidualSubspaceExpertsEmbed(
            VOCAB_SIZE,
            EMBED_DIM,
            base_rank=BASE_RANK,
            expert_rank=EXPERT_RANK,
            num_experts=NUM_EXPERTS,
            router_dim=ROUTER_DIM,
            top_k=TOP_K,
            router_temperature=ROUTER_TEMPERATURE,
        )
        self.head = make_tied_head(
            self.embed, "residual_subspace_experts", VOCAB_SIZE
        )

    def forward(self, input_ids, labels):
        # Calling the input embedder before the head is important: production
        # training shares this exact vocabulary route between the two sides.
        hidden, _ = self.embed(input_ids)
        logits = self.head(hidden)
        language_loss = F.cross_entropy(
            logits.float().reshape(-1, VOCAB_SIZE), labels.reshape(-1)
        )
        router_loss = self.embed.pop_router_aux_loss()
        if router_loss is None:
            raise AssertionError("training forward did not produce router loss")
        return language_loss + 0.01 * router_loss.float()


def build_model(device):
    return SmokeModel().to(device=device, dtype=torch.bfloat16)


def assert_finite_gradients(model):
    missing = []
    nonfinite = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all().item():
            nonfinite.append(name)
    if missing:
        raise AssertionError(f"parameters outside autograd: {missing}")
    if nonfinite:
        raise AssertionError(f"non-finite gradients: {nonfinite}")


def one_training_step(model, optimizer, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    input_ids = torch.randint(
        VOCAB_SIZE, (2, 4), generator=generator, device=device
    )
    labels = torch.randint(
        VOCAB_SIZE, (2, 4), generator=generator, device=device
    )
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids, labels)
    if not torch.isfinite(loss).item():
        raise AssertionError(f"non-finite loss: {loss.item()}")
    loss.backward()
    assert_finite_gradients(model)
    optimizer.step()
    return float(loss.detach())


def check_sampled_tying(model, device):
    model.eval()
    sample_ids = torch.tensor(
        [0, 1, 7, 127, 1_023, 8_191, 65_535, VOCAB_SIZE - 1],
        device=device,
    )
    with torch.no_grad():
        table = model.embed.materialize(sample_ids)
        hidden = torch.randn(3, EMBED_DIM, device=device, dtype=torch.bfloat16)
        factored = model.head(hidden).index_select(-1, sample_ids)
        materialized = hidden @ table.T
    torch.testing.assert_close(
        factored, materialized, rtol=3e-2, atol=3e-2
    )
    return float((factored.float() - materialized.float()).abs().max())


def test_each_visible_gpu(world_size):
    print("\n=== Per-GPU production-shape architecture test ===", flush=True)
    for device_index in range(world_size):
        device = torch.device("cuda", device_index)
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(10_000 + device_index)

        started = time.monotonic()
        model = build_model(device)
        expected_parameters = 19_471_968
        actual_parameters = sum(p.numel() for p in model.parameters())
        if actual_parameters != expected_parameters:
            raise AssertionError(
                f"parameter count {actual_parameters} != {expected_parameters}"
            )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = [
            one_training_step(
                model, optimizer, device, 20_000 + 10 * device_index + step
            )
            for step in range(2)
        ]
        max_tied_error = check_sampled_tying(model, device)
        torch.cuda.synchronize(device)
        peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
        elapsed = time.monotonic() - started
        print(
            f"GPU {device_index}: PASS | losses={losses} | "
            f"sampled_tied_max_abs_error={max_tied_error:.6g} | "
            f"peak_allocated={peak_gib:.2f} GiB | elapsed={elapsed:.1f}s",
            flush=True,
        )
        del optimizer, model
        torch.cuda.empty_cache()


def ddp_worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=dt.timedelta(minutes=5),
    )
    try:
        device = torch.device("cuda", rank)
        torch.manual_seed(30_000 + rank)
        model = build_model(device)
        ddp = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
        losses = [
            one_training_step(
                ddp, optimizer, device, 40_000 + 100 * rank + step
            )
            for step in range(2)
        ]

        # DDP broadcasts initialization and all-reduces gradients. Identical
        # post-step parameter signatures verify that reduction/update occurred
        # consistently on every rank.
        signature = torch.stack([
            parameter.detach().reshape(-1)[0].float()
            for parameter in ddp.module.parameters()
        ])
        gathered = [torch.empty_like(signature) for _ in range(world_size)]
        dist.all_gather(gathered, signature)
        max_difference = max(
            float((candidate - gathered[0]).abs().max())
            for candidate in gathered
        )
        if max_difference > 1e-6:
            raise AssertionError(
                f"DDP parameter signatures diverged by {max_difference}"
            )

        # This explicit scalar collective complements the gradient all-reduces
        # above and makes a broken NCCL process group fail deterministically.
        marker = torch.tensor(float(rank + 1), device=device)
        dist.all_reduce(marker)
        expected_marker = world_size * (world_size + 1) / 2
        if marker.item() != expected_marker:
            raise AssertionError(
                f"NCCL all-reduce returned {marker.item()}, "
                f"expected {expected_marker}"
            )
        dist.barrier()
        if rank == 0:
            print(
                f"DDP rank 0: local losses={losses} | "
                f"signature_max_difference={max_difference:.3g} | "
                f"NCCL marker={marker.item():.0f}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def free_tcp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def parse_args():
    parser = argparse.ArgumentParser()
    default_log = REPO_ROOT / "temp" / (
        "residual_subspace_experts_gpu_"
        + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".log"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=default_log,
        help="output log (default: temp/residual_subspace_experts_gpu_<time>.log)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("w", encoding="utf-8", buffering=1) as log:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(old_stdout, log)
        sys.stderr = Tee(old_stderr, log)
        try:
            print(f"log_file: {args.log_file.resolve()}")
            print(f"python: {sys.executable}")
            print(f"torch: {torch.__version__}")
            print(f"torch_cuda_build: {torch.version.cuda}")
            print(f"cuda_available: {torch.cuda.is_available()}")
            world_size = torch.cuda.device_count()
            print(f"visible_gpu_count: {world_size}")
            if not torch.cuda.is_available() or world_size == 0:
                raise RuntimeError("PyTorch cannot access CUDA")
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("visible CUDA device does not support BF16")
            for index in range(world_size):
                props = torch.cuda.get_device_properties(index)
                print(
                    f"GPU {index}: {props.name} | "
                    f"memory={props.total_memory / 2**30:.1f} GiB | "
                    f"capability={props.major}.{props.minor}"
                )

            test_each_visible_gpu(world_size)
            if world_size < 2:
                raise RuntimeError(
                    "DDP test requires at least two visible GPUs; "
                    f"found {world_size}"
                )
            print(
                f"\n=== {world_size}-GPU NCCL/DDP test "
                "(find_unused_parameters=False) ===",
                flush=True,
            )
            started = time.monotonic()
            mp.spawn(
                ddp_worker,
                args=(world_size, free_tcp_port()),
                nprocs=world_size,
                join=True,
            )
            print(
                f"DDP: PASS | ranks={world_size} | "
                f"elapsed={time.monotonic() - started:.1f}s"
            )
            print("\nRESULT: PASS")
        except BaseException:
            print("\nRESULT: FAIL")
            traceback.print_exc()
            raise
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if __name__ == "__main__":
    main()
