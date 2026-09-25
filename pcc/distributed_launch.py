"""Offline single-node torchrun launch and worker lifecycle."""
from datetime import timedelta
import os
import subprocess
import sys

from .joint_config import settings_for


def execute(args, config):
    gpus = args.physical_gpus
    settings = settings_for(config)
    if len(gpus) < 2 or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        raise ValueError("Distributed launch requires distinct nonnegative GPU indices")
    contexts = settings.tokens_per_update // settings.context
    if contexts % len(gpus) or (contexts // len(gpus)) % config["microbatch"]:
        raise ValueError("Per-rank microbatch must divide the fixed global context count")
    if settings.version == "joint-local-v3" and gpus != list(range(4)):
        raise ValueError("joint-local-v3 requires all four local GPUs")
    if settings.version == "joint-v2-ddp" and len(gpus) != 8:
        raise ValueError("joint-v2-ddp requires eight GPUs per experiment")
    from .__main__ import configure_offline
    configure_offline()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    if "LOCAL_RANK" not in os.environ:
        from scripts.gpu_status import require_free
        require_free(gpus)
        if args.output.exists():
            raise ValueError("Output exists; distributed runs require a fresh directory")
        subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                        "--nnodes=1", f"--nproc-per-node={len(gpus)}", "--max-restarts=0",
                        "-m", "pcc.joint", *sys.argv[1:]], check=True)
        return
    import torch
    import torch.distributed as dist
    if int(os.environ["WORLD_SIZE"]) != len(gpus):
        raise ValueError("torchrun world size differs from declared allocation")
    local_rank = int(os.environ["LOCAL_RANK"])
    if not 0 <= local_rank < len(gpus) or int(os.environ["RANK"]) != local_rank:
        raise ValueError("Only a single-node rank mapping is supported")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10),
                            device_id=torch.device("cuda", local_rank))
    try:
        if local_rank == 0 and args.output.exists():
            raise ValueError("Output exists; use a fresh output directory")
        dist.barrier()
        from .joint import run_arm
        report = run_arm(args, config)
        if local_rank == 0:
            import json
            print(json.dumps(report, indent=2), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()
