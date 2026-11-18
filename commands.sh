#1 +60+a
#th2-ohmv-route-gradient-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_route_gradient --commit 791112b8ff2ea8a060400d3af4bdca08ec5a5559 --prepared-sha256 4d7d8046351448b65dde3bca87654ca25b55aae2c02cd0f55fa5a232fc381d5b
