#1 +60+a
#th2-ohmv-window-film-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_window_film --commit 4f95ed5d9bf9bd7bc2b1575e5cfa8f6ce7007230 --prepared-sha256 82a162607bf1556a7cfb0736f5e751212bb1d0a7c6f305f42393edbc6d9ce85b
