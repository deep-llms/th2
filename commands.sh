#1 +60+a
#th2-ohmv-auxiliary-order-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_auxiliary --commit b4bf298928278a6852857104ef7fd4784801f21c --prepared-sha256 338fa145bcb4f0cc4879cc478a06dcb397e65dce1605bbe5bc3e77e8788bae9b
