#1 +60+a
#th2-ohmv-exposure-recovery-launch-20261119-a02
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_exposure_recovery --commit 0b71555da9ad0e79aabc61d440b1907a2c6bc79c --prepared-sha256 89bac222c5804789d5b85d930e1a2f521fcca018708fd8b5aef7aed982790b50
