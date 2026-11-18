#1 +60+a
#th2-ohmv-token-order-launch-20261118-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_token_order --commit aeb4fdd95b12177ae2a0b7e3ae7019153d66b904 --prepared-sha256 23d97c3ca6ff1be77d812c9af7e21b8f407f98d866dcfcd3250287c6cd216a1f
