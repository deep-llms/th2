#1 +30+a
#th2-q359-dot-preflight-20261018-a03
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1
python3 scripts/bdot_b200_preflight.py --output /mnt/local/_outputs/deep-llms_th2/q359-dot-verification-20261018-a03/preflight.json
