#1 +60+a
#th2-ohmv-window-recovery-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_window_recovery --commit d4f484249b0a2616ff05292791de3c948b3e12c8 --prepared-sha256 25a15df94070559bd11fcedaae40fa940ecbcc7a61143a63b7b799bcc83e6da4
