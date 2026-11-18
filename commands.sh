#1 +60+a
#th2-ohmv-balance-units-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_balance_units --commit b291c71f989142b676476f8f4894856ef08bb053 --prepared-sha256 bec6b653267008881cd51c5eeb4854ce1c444bae5baa47cbd59ceb4d50364b88
