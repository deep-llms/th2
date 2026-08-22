#!/usr/bin/env python3
"""Saturate one CUDA device with repeated dense matrix multiplications."""

import argparse

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-size", type=int, default=16384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.matrix_size <= 0:
        raise ValueError("--matrix-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Each burn worker must see exactly one CUDA device; "
            f"found {torch.cuda.device_count()}"
        )

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    left = torch.randn(
        (args.matrix_size, args.matrix_size), device=device, dtype=dtype
    )
    right = torch.randn(
        (args.matrix_size, args.matrix_size), device=device, dtype=dtype
    )
    output = torch.empty_like(left)
    torch.cuda.synchronize(device)

    print(
        "gpu_burn_ready",
        f"device={torch.cuda.get_device_name(device)}",
        f"matrix_size={args.matrix_size}",
        f"dtype={dtype}",
        flush=True,
    )

    while True:
        torch.mm(left, right, out=output)
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
