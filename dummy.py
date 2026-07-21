"""Occupy GPUs with matrix multiplications. Keeps GPUs busy until killed.

Allocates ~110GB per GPU and runs continuous matmuls so GPU utilization
and power draw are high.

Usage:
    python dummy.py                  # all GPUs
    python dummy.py --gpus 0 1 2 3   # specific GPUs

Kill with Ctrl+C or: pkill -f dummy.py
"""

import argparse
import signal
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", type=int, default=None)
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    gpus = args.gpus or list(range(n_gpus))

    matrices = []
    for gpu_id in gpus:
        tensors = []
        allocated = 0
        while allocated < 110:
            try:
                t = torch.randn(8192, 8192, device=f"cuda:{gpu_id}", dtype=torch.float32)
                tensors.append(t)
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
            except torch.cuda.OutOfMemoryError:
                break
        matrices.append((gpu_id, tensors))
        print(f"GPU {gpu_id}: {allocated:.1f} GB allocated")

    print(f"\nRunning matmuls on {len(gpus)} GPUs. Ctrl+C or pkill -f dummy.py to stop.")

    stop = False
    def handle_signal(sig, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while not stop:
            for gpu_id, tensors in matrices:
                torch.mm(tensors[0], tensors[1], out=tensors[0])
    except KeyboardInterrupt:
        pass

    del matrices
    torch.cuda.empty_cache()
    print("Released.")


if __name__ == "__main__":
    main()
