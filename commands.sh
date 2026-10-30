#1 +30+a
#th2-6y3o-a2-quant-20261031-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_quant --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01 --session a2_quant_20261031_a01 --code-commit 422bb7f118fd875ea2d9392e5da0cefbfbfb8d30
