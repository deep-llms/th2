#1 +60+a
#th2-ohmv-order-distill-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_order_distill --commit 76605f210f011380d8da43dccf3d5316e041e3dc --prepared-sha256 553ee55fbce9772a9ee9a406fa21aa7157459f4a1f9a4cbbc5e6137ebbe33f4e
