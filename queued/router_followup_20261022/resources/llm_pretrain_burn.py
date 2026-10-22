#!/usr/bin/env python3
"""Training-like CUDA burn with bounded high-memory occupancy."""

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# This default is intentionally self-contained: no environment variable is
# required. Three persistent 8192-square BF16/FP16 matrices use 384 MiB.
MATRIX_SIZE = int(os.environ.get("GPU_BURN_MATRIX_SIZE", "8192"))
# Resolve the memory target independently from each GPU's reported capacity.
TARGET_MEMORY_FRACTION = float(
    os.environ.get("GPU_BURN_MEMORY_FRACTION", "0.85")
)
MIN_FREE_GIB = float(os.environ.get("GPU_BURN_MIN_FREE_GIB", "8"))
# Qwen3-0.6B has 596,049,920 unique parameters. Its BF16 DDP gradient
# payload is 1,136.875 MiB, rounded up to a whole MiB here.
COMM_TOTAL_MIB = int(os.environ.get("GPU_BURN_COMM_TOTAL_MIB", "1137"))
COMM_BUCKET_MIB = int(os.environ.get("GPU_BURN_COMM_BUCKET_MIB", "25"))
APPROX_STEP_SECONDS = float(
    os.environ.get("GPU_BURN_APPROX_STEP_SECONDS", "0.75")
)
CALIBRATION_GEMMS = int(os.environ.get("GPU_BURN_CALIBRATION_GEMMS", "64"))
PROGRESS_EVERY = int(os.environ.get("GPU_BURN_PROGRESS_EVERY", "10"))
MIN_WORLD_SIZE = int(os.environ.get("GPU_BURN_MIN_WORLD_SIZE", "2"))
GIB = 1 << 30
MIB = 1 << 20
RESERVE_ALIGNMENT = 256 * MIB


def used_memory(device: torch.device) -> tuple[int, int, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return total_bytes - free_bytes, free_bytes, total_bytes


def all_reduce_buckets(buffer: torch.Tensor, bucket_elements: int) -> None:
    for bucket in buffer.split(bucket_elements):
        dist.all_reduce(bucket, op=dist.ReduceOp.SUM)


def training_like_cycle(
    left: torch.Tensor,
    right: torch.Tensor,
    output: torch.Tensor,
    comm_buffer: torch.Tensor,
    bucket_elements: int,
    gemm_count: int,
    world: int,
) -> None:
    """Spread DDP-sized bucket reductions across one compute cycle."""
    buckets = comm_buffer.split(bucket_elements)
    base_gemms, extra_gemms = divmod(gemm_count, len(buckets))
    for bucket_index, bucket in enumerate(buckets):
        bucket_gemms = base_gemms + (bucket_index < extra_gemms)
        for _ in range(bucket_gemms):
            torch.mm(left, right, out=output)
        if world > 1:
            dist.all_reduce(bucket, op=dist.ReduceOp.SUM)
            bucket.mul_(1.0 / world)


def burn(rank: int, world: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if world > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
    minimum_free_bytes = int(MIN_FREE_GIB * GIB)
    element_size = torch.empty((), dtype=dtype).element_size()
    comm_elements = COMM_TOTAL_MIB * MIB // element_size
    bucket_elements = COMM_BUCKET_MIB * MIB // element_size
    comm_bucket_count = (
        COMM_TOTAL_MIB + COMM_BUCKET_MIB - 1
    ) // COMM_BUCKET_MIB

    with torch.no_grad():
        collective_probe = torch.tensor(rank + 1, device=device, dtype=torch.int64)
        if world > 1:
            dist.all_reduce(collective_probe, op=dist.ReduceOp.SUM)
        expected_probe_sum = world * (world + 1) // 2
        if collective_probe.item() != expected_probe_sum:
            raise RuntimeError(
                f"NCCL collective check failed on rank {rank}: got "
                f"{collective_probe.item()}, expected {expected_probe_sum}"
            )

        left = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device, dtype=dtype)
        right = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device, dtype=dtype)
        output = torch.empty_like(left)
        comm_buffer = torch.randn(comm_elements, device=device, dtype=dtype)

        # Warm up and measure the persistent compute and communication operations.
        torch.mm(left, right, out=output)
        torch.cuda.synchronize(device)
        gemm_start = time.monotonic()
        for _ in range(CALIBRATION_GEMMS):
            torch.mm(left, right, out=output)
        torch.cuda.synchronize(device)
        gemm_seconds = time.monotonic() - gemm_start
        seconds_per_gemm = gemm_seconds / CALIBRATION_GEMMS

        if world > 1:
            # First pass initializes every NCCL bucket path; the second is timed.
            all_reduce_buckets(comm_buffer, bucket_elements)
            comm_buffer.mul_(1.0 / world)
        torch.cuda.synchronize(device)
        comm_start = time.monotonic()
        if world > 1:
            all_reduce_buckets(comm_buffer, bucket_elements)
            comm_buffer.mul_(1.0 / world)
        torch.cuda.synchronize(device)
        comm_seconds = time.monotonic() - comm_start

        target_compute_seconds = max(
            seconds_per_gemm, APPROX_STEP_SECONDS - comm_seconds
        )
        gemms_per_sync = max(1, round(target_compute_seconds / seconds_per_gemm))
        if world > 1:
            calibrated = torch.tensor([gemms_per_sync], device=device, dtype=torch.int64)
            dist.broadcast(calibrated, src=0)
            gemms_per_sync = int(calibrated.item())

        current_used, current_free, total = used_memory(device)
        target_used_bytes = int(total * TARGET_MEMORY_FRACTION)
        if target_used_bytes >= total - minimum_free_bytes:
            raise RuntimeError(
                f"target fraction {TARGET_MEMORY_FRACTION:.1%} leaves less than "
                f"{MIN_FREE_GIB:.2f} GiB free on a {total / GIB:.2f} GiB GPU"
            )
        reserve_bytes = target_used_bytes - current_used
        reserve_bytes = (reserve_bytes // RESERVE_ALIGNMENT) * RESERVE_ALIGNMENT
        if reserve_bytes <= 0:
            raise RuntimeError(
                f"current usage {current_used / GIB:.2f} GiB already meets or "
                f"exceeds the {TARGET_MEMORY_FRACTION:.1%} target "
                f"({target_used_bytes / GIB:.2f} GiB)"
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
        if abs(actual_used - target_used_bytes) > RESERVE_ALIGNMENT:
            raise RuntimeError(
                f"actual usage {actual_used / GIB:.2f} GiB is not within "
                f"{RESERVE_ALIGNMENT / MIB:.0f} MiB of the "
                f"{target_used_bytes / GIB:.2f} GiB target"
            )
        if actual_free < minimum_free_bytes:
            raise RuntimeError(
                f"actual free memory {actual_free / GIB:.2f} GiB is below "
                f"the {MIN_FREE_GIB:.2f} GiB safety floor"
            )
        print(
            "gpu_burn_ready",
            f"rank={rank}",
            f"device={torch.cuda.get_device_name(device)}",
            f"matrix_size={MATRIX_SIZE}",
            f"dtype={dtype}",
            f"target_memory_fraction={TARGET_MEMORY_FRACTION:.4f}",
            f"target_used_gib={target_used_bytes / GIB:.2f}",
            f"actual_used_gib={actual_used / GIB:.2f}",
            f"actual_memory_fraction={actual_used / total:.4f}",
            f"actual_free_gib={actual_free / GIB:.2f}",
            f"total_gib={total / GIB:.2f}",
            f"reserve_gib={memory_reserve.numel() / GIB:.2f}",
            f"world_size={world}",
            f"collective_probe_sum={collective_probe.item()}",
            f"comm_total_mib={COMM_TOTAL_MIB}",
            f"comm_bucket_mib={COMM_BUCKET_MIB}",
            f"comm_bucket_count={comm_bucket_count}",
            "communication_pattern=interleaved_with_compute",
            f"ring_equivalent_send_mib_per_rank="
            f"{2 * (world - 1) * COMM_TOTAL_MIB / world:.2f}",
            f"approx_step_seconds={APPROX_STEP_SECONDS:.3f}",
            f"calibration_gemms={CALIBRATION_GEMMS}",
            f"seconds_per_gemm={seconds_per_gemm:.6f}",
            f"measured_comm_seconds={comm_seconds:.6f}",
            f"gemms_per_sync={gemms_per_sync}",
            flush=True,
        )

        completed_cycles = 0
        progress_start = time.monotonic()
        while True:
            training_like_cycle(
                left,
                right,
                output,
                comm_buffer,
                bucket_elements,
                gemms_per_sync,
                world,
            )
            torch.cuda.synchronize(device)
            completed_cycles += 1
            if rank == 0 and completed_cycles % PROGRESS_EVERY == 0:
                elapsed = time.monotonic() - progress_start
                print(
                    "gpu_burn_progress",
                    f"rank={rank}",
                    f"completed_cycles={completed_cycles}",
                    f"completed_gemms={completed_cycles * gemms_per_sync}",
                    f"completed_collective_payload_gib="
                    f"{completed_cycles * COMM_TOTAL_MIB / 1024:.2f}",
                    f"average_cycle_seconds={elapsed / completed_cycles:.3f}",
                    f"cycles_per_second={completed_cycles / elapsed:.3f}",
                    flush=True,
                )


def main() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    # Match this project's DDP training launchers unless explicitly overridden.
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    world = torch.cuda.device_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<all>")
    print(
        f"GPU burn: {world} visible GPU(s) "
        f"(CUDA_VISIBLE_DEVICES={visible}, "
        f"target_memory_fraction={TARGET_MEMORY_FRACTION:.1%})",
        flush=True,
    )
    if world <= 0:
        raise SystemExit("no visible CUDA devices")
    if MIN_WORLD_SIZE <= 0:
        raise SystemExit("GPU_BURN_MIN_WORLD_SIZE must be positive")
    if world < MIN_WORLD_SIZE:
        raise SystemExit(
            f"need at least {MIN_WORLD_SIZE} visible GPUs for the requested "
            f"burn, found {world}"
        )
    if MATRIX_SIZE <= 0:
        raise SystemExit("GPU_BURN_MATRIX_SIZE must be positive")
    if not (0 < TARGET_MEMORY_FRACTION < 1):
        raise SystemExit("GPU_BURN_MEMORY_FRACTION must be between 0 and 1")
    if not (0 < MIN_FREE_GIB):
        raise SystemExit("GPU_BURN_MIN_FREE_GIB must be positive")
    if COMM_TOTAL_MIB <= 0:
        raise SystemExit("GPU_BURN_COMM_TOTAL_MIB must be positive")
    if COMM_BUCKET_MIB <= 0:
        raise SystemExit("GPU_BURN_COMM_BUCKET_MIB must be positive")
    if APPROX_STEP_SECONDS <= 0:
        raise SystemExit("GPU_BURN_APPROX_STEP_SECONDS must be positive")
    if CALIBRATION_GEMMS <= 0:
        raise SystemExit("GPU_BURN_CALIBRATION_GEMMS must be positive")
    if PROGRESS_EVERY <= 0:
        raise SystemExit("GPU_BURN_PROGRESS_EVERY must be positive")

    if world == 1:
        burn(0, 1)
    else:
        mp.spawn(burn, args=(world,), nprocs=world)


if __name__ == "__main__":
    main()
