#1 +60+a
#th2-ohmv-window-cf-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_window_cf --commit 2fb90f1ad6ff4583addcd4de17fab7812dbe1e4c --prepared-sha256 8cfc742a3f8af73903e153cd139ede366b0e4d8c415276c5ba9ff371ad6c34ff
