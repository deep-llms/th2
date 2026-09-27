#!/bin/bash
# Standard HF arguments or one JSON config, exactly as accepted by train.py.
set -euo pipefail
export WANDB_PROJECT="deep2shallow" WANDB_MODE=offline NCCL_NVLS_ENABLE=0
exec python -m accelerate.commands.launch \
    --config_file resources/accelerate_config.yaml train.py "$@"
