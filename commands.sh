#1 +60+a
#th2-ohmv-frozen-router-launch-20261120-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_frozen_router --commit c236ebba6018a88fcc6a00bd66e3a431eb89e329 --prepared-sha256 4bd3dc5efc7024f126d3f3192e0a49c242f011616e9398cdc54048e5e084e58b
