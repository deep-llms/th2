#1 +30+a
#th2-6y3o-attention-compare-20261020-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline HF_HUB_DISABLE_TELEMETRY=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 NCCL_NVLS_ENABLE=0
/mnt/local/conda-py311/envs/train/bin/python3.11 scripts/bdot_b200_launch.py --root /mnt/local/_outputs/deep-llms_th2/6y3o-attention-compare-20261020-a01 --host thiennh-p6-6y3o-worker-0 --session attention_compare_20261020_a01 --verify-python /mnt/local/conda-py311/envs/train_fa2_t28/bin/python3.11 --jobs-config resources/attention_comparison_20261020.json
