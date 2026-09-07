"""Tiny CPU/CUDA/DDP smoke gate. Does not stop any other workloads.

python -m scripts.smoke_capacity --output temp/capacity_smoke.json
torchrun --standalone --nproc_per_node=2 -m scripts.smoke_capacity --output temp/ddp.json
For authorized GPU tests only, add --device cuda after verifying free GPUs.
"""
import argparse
import json
import math
import os
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
    args = parser.parse_args()
    torch.set_num_threads(2)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected GPU does not support bf16")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    results = []
    try:
        for arm in ARMS:
            set_seed(17)
            model = build_model(experiment_config(arm, tiny=True)).to(device)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            counts = parameter_report(model)
            wrapped = (DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None,
                                               find_unused_parameters=False) if world > 1 else model)
            optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
            generator = torch.Generator(device=device).manual_seed(100 + rank)
            losses = []
            for step in range(3):
                x = torch.randint(0, 97, (2, 12), device=device, generator=generator)
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
            if world > 1:
                # Compare every parameter, not merely a checksum, after different
                # rank-local batches. This also checks actual collective execution.
                for parameter in model.parameters():
                    reference = parameter.detach().clone()
                    dist.broadcast(reference, src=0)
                    torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
            scales = activation_report(model, x)
            if not all(math.isfinite(v) for v in scales.values()):
                raise AssertionError(f"{arm}: bad activation scales")
            results.append(dict(arm=arm, losses=losses, parameters=counts, final_scales=scales))
            del optimizer, wrapped, model
        if rank == 0:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, dict(result="PASS", world_size=world, device=str(device),
                                  precision=args.precision, tiny_models=True, experiments=results))
            print(json.dumps(dict(result="PASS", output=str(path), ranks=world)), flush=True)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
