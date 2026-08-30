#!/usr/bin/env python3
"""Training-like CUDA burn with bounded high-memory occupancy."""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


MATRIX_SIZE = int(os.environ.get("GPU_BURN_MATRIX_SIZE", "8192"))
TARGET_USED_GIB = float(os.environ.get("GPU_BURN_TARGET_GIB", "165"))
MIN_FREE_GIB = float(os.environ.get("GPU_BURN_MIN_FREE_GIB", "8"))
GIB = 1 << 30
MIB = 1 << 20
RESERVE_ALIGNMENT = 256 * MIB


def used_memory(device: torch.device) -> tuple[int, int, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return total_bytes - free_bytes, free_bytes, total_bytes


def burn(rank: int, world: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if world > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
    target_used_bytes = int(TARGET_USED_GIB * GIB)
    minimum_free_bytes = int(MIN_FREE_GIB * GIB)

    with torch.no_grad():
        left = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device, dtype=dtype)
        right = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device, dtype=dtype)
        output = torch.empty_like(left)

        # Warm up every persistent CUDA/NCCL operation before sizing the reserve.
        torch.mm(left, right, out=output)
        if world > 1:
            token = output.sum().reshape(1)
            dist.all_reduce(token, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)

        current_used, current_free, total = used_memory(device)
        if target_used_bytes >= total - minimum_free_bytes:
            raise RuntimeError(
                f"target {TARGET_USED_GIB:.2f} GiB leaves less than "
                f"{MIN_FREE_GIB:.2f} GiB free on a {total / GIB:.2f} GiB GPU"
            )
        reserve_bytes = target_used_bytes - current_used
        reserve_bytes = (reserve_bytes // RESERVE_ALIGNMENT) * RESERVE_ALIGNMENT
        if reserve_bytes <= 0:
            raise RuntimeError(
                f"current usage {current_used / GIB:.2f} GiB already meets or "
                f"exceeds target {TARGET_USED_GIB:.2f} GiB"
            )
        if reserve_bytes > current_free - minimum_free_bytes:
            raise RuntimeError(
                f"cannot reserve {reserve_bytes / GIB:.2f} GiB while retaining "
                f"{MIN_FREE_GIB:.2f} GiB free"
            )

        # Retain and touch the allocation so the requested HBM is committed.
        memory_reserve = torch.empty(reserve_bytes, device=device, dtype=torch.uint8)
        memory_reserve.zero_()
        torch.cuda.synchronize(device)

        actual_used, actual_free, total = used_memory(device)
        print(
            "gpu_burn_ready",
            f"rank={rank}",
            f"device={torch.cuda.get_device_name(device)}",
            f"matrix_size={MATRIX_SIZE}",
            f"dtype={dtype}",
            f"target_used_gib={TARGET_USED_GIB:.2f}",
            f"actual_used_gib={actual_used / GIB:.2f}",
            f"actual_free_gib={actual_free / GIB:.2f}",
            f"total_gib={total / GIB:.2f}",
            f"reserve_gib={memory_reserve.numel() / GIB:.2f}",
            flush=True,
        )

        while True:
            torch.mm(left, right, out=output)
            if world > 1:
                token = output.sum().reshape(1)
                dist.all_reduce(token, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device)


def main() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    world = torch.cuda.device_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<all>")
    print(
        f"GPU burn: {world} visible GPU(s) "
        f"(CUDA_VISIBLE_DEVICES={visible}, target_used_gib={TARGET_USED_GIB:.2f})",
        flush=True,
    )
    if world <= 0:
        raise SystemExit("no visible CUDA devices")
    if MATRIX_SIZE <= 0:
        raise SystemExit("GPU_BURN_MATRIX_SIZE must be positive")
    if not (0 < TARGET_USED_GIB):
        raise SystemExit("GPU_BURN_TARGET_GIB must be positive")
    if not (0 < MIN_FREE_GIB):
        raise SystemExit("GPU_BURN_MIN_FREE_GIB must be positive")

    if world == 1:
        burn(0, 1)
    else:
        mp.spawn(burn, args=(world,), nprocs=world)


if __name__ == "__main__":
    main()
