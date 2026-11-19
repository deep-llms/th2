#1 +60+a
#th2-ohmv-horizon-warmup-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_horizon --commit 69daf5aa46e70470df2d17b171789cabf09136d0 --prepared-sha256 215157dab4033a4e19273010f2e0262be94c97037a915233c0bc954019f7eafa
