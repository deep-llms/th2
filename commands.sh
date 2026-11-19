#1 +60+a
#th2-ohmv-progressive-warmup-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_progressive --commit 9ddf731218fc89635eed3a087821a3178577f3ae --prepared-sha256 62ae85e460d70a926915e5be765bbfbccffa10f0fe063d9cf5a240e15c6021b1
