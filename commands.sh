#1 +60+a
#th2-ohmv-soft-route-launch-20261119-a02
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_soft_route --commit f3df076057411deadce48df043c8697c4b1efa87 --prepared-sha256 37925ed89422804e8e1e8276e3ccf9e56ba45ef597a77bc6991970c3d7d2a5c9
