#1 +60+a
#th2-6y3o-a2-mbs8-20261029-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_performance \
 --suite mbs8 --inputs /mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs \
 --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-mbs8-20261029-a01 \
 --predecessor-burn-script /mnt/local/_outputs/deep-llms_th2/6y3o-a2-performance-20261029-a01/source/resources/llm_pretrain_burn.py \
 --host thiennh-p6-6y3o-worker-0 --session a2_mbs8_20261029_a01 \
 --code-commit 04ab56618eae019b13aec0ba8fd784b1937faed8
