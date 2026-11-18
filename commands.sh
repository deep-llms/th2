#1 +30+a
#th2-ohmv-warm-order-health-20261118-a04
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.warm_order_health
