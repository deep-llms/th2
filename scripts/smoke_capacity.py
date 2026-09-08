"""Tiny CPU/CUDA/DDP smoke gate. Does not stop any other workloads.

python -m scripts.smoke_capacity --output temp/capacity_smoke.json
torchrun --standalone --nproc_per_node=2 -m scripts.smoke_capacity --output temp/ddp.json
For authorized GPU tests only, add --device cuda after verifying free GPUs.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import set_seed

from capacity_allocation.data import write_json
from capacity_allocation.modeling import ARMS, activation_report, build_model, experiment_config, parameter_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--output", required=True)
    parser.add_argument("--production", action="store_true", help="Actual six-layer Qwen dimensions; CUDA only")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(2)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device("cpu")
    if args.batch_size <= 0 or args.sequence_length < 2:
        parser.error('Positive batch size and sequence length >= 2 required')
    if args.production and (device.type != 'cuda' or args.sequence_length != 2048):
        parser.error('Production smoke requires CUDA and sequence length 2048')
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected GPU does not support bf16")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    results = []
    try:
        for arm in args.arms:
            set_seed(17)
            model = build_model(experiment_config(arm, tiny=not args.production)).to(device)
            if not args.production:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            counts = parameter_report(model)
            wrapped = (DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None,
                                               find_unused_parameters=False) if world > 1 else model)
            optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
            generator = torch.Generator(device=device).manual_seed(100 + rank)
            losses, durations = [], []
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            for step in range(3):
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                started = time.monotonic()
                x = torch.randint(0, model.config.vocab_size, (args.batch_size, args.sequence_length), device=device, generator=generator)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                    loss = wrapped(x, labels=x, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise AssertionError(f"{arm}: non-finite loss")
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise AssertionError(f"{arm}: missing or nonfinite gradient in {name}")
                optimizer.step()
                losses.append(loss.item())
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                durations.append(time.monotonic()-started)
            if world > 1:
                # Compare every parameter, not merely a checksum, after different
                # rank-local batches. This also checks actual collective execution.
                for parameter in model.parameters():
                    reference = parameter.detach().clone()
                    dist.broadcast(reference, src=0)
                    torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                scales = activation_report(model, x[:1, :128])
            if not all(math.isfinite(v) for v in scales.values()):
                raise AssertionError(f"{arm}: bad activation scales")
            peak_gib = torch.cuda.max_memory_allocated(device)/(1 << 30) if device.type == 'cuda' else None
            result = dict(arm=arm, losses=losses, parameters=counts, final_scales=scales,
                          step_seconds=durations, peak_allocated_gib=peak_gib)
            print(json.dumps(dict(rank=rank, **result)), flush=True)
            results.append(result)
            del optimizer, wrapped, model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        if rank == 0:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, dict(result="PASS", world_size=world, device=str(device),
                                  precision=args.precision, tiny_models=not args.production,
                                  batch_size=args.batch_size, sequence_length=args.sequence_length, experiments=results))
            print(json.dumps(dict(result="PASS", output=str(path), ranks=world)), flush=True)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
