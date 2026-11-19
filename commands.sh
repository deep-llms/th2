#1 +60+a
#th2-ohmv-soft-route-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_soft_route --commit 6dd95e7ad7b0de0710458a27f1ea692f38943187 --prepared-sha256 5e336a63a5cfd5e578ca3ec4336c6d51556f0a25bce4fb20e4ede4135f1f881a
