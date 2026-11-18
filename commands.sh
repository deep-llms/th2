#1 +60+a
#th2-ohmv-adaptive-refinement-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_adaptive_refinement --commit 22ebaf732ce72e9f51928d19c2d1b653ef0d6ecc --prepared-sha256 daf94a336046e9f9c2583d45386f59ee3da2cad88834b5f5a7f86a68b0802f1b
