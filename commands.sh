#1 +20+a
#th2-wc69-family-f9-v110-launch-20261110-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_family_f9_v110 --plan resources/a2_family_f9_v110_plan_20261110.json --output /mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-v110-20261110-a01 --session a2_family_f9_v110_20261110_a01 --code-commit f9e097b848649295104c6448cf451389fdb3382e
