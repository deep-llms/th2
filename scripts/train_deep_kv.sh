#!/bin/bash
# Launch one Deep-KV arm using the baseline's eight-GPU Accelerate/HF workflow.
# Config selects offline assets and microbatch 16; Trainer accumulates four times.
# Pass --config, --data-dir, --output, --arm and optional --stop-after/--resume.
set -euo pipefail
export WANDB_PROJECT="deep2shallow"
export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
exec python -m accelerate.commands.launch \
    --multi_gpu --num_machines 1 --num_processes 8 --mixed_precision bf16 \
    --dynamo_backend no --module deep_kv train "$@"
