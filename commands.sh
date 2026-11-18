#1 +60+a
#th2-ohmv-order-diagnostics-launch-20261118-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_order_diagnostics --commit 9ba4b6128ef4055cb4777130d7f1c6ad28e10725 --prepared-sha256 60d5818d228f93f9853808858b4b1519f7c09f5139d2975c37d2d089fd0e995b
