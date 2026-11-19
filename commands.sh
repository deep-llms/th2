#1 +60+a
#th2-ohmv-soft-exploration-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_soft_exploration --commit 9158b0ba0e28cef4bf7c7f7c653455c658fb7291 --prepared-sha256 0118b9eb0bd6302fe507f7fc92532f78a26156322c04be2e9aa6f3600c9db7e7
