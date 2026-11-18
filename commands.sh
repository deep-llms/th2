#1 +60+a
#th2-ohmv-window-order-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_window_order --commit 031972d3f317fa3b3632947555ce70cab1cefbad --prepared-sha256 dce481b8dcec8bd47c552025c299f25f5ee879b8859366381c452da5f86ea423
