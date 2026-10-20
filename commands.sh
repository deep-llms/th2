#1 +30+a
#th2-6y3o-attention-study-preflight-20261020-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
python3 scripts/attention_env_inventory.py --host thiennh-p6-6y3o-worker-0 --require-absent train_fa2_t28 --output /mnt/local/_outputs/deep-llms_th2/6y3o-attention-study-20261020-a01/inventory-before.json
