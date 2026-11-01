#1 +20+a
#th2-6y3o-a2-depth-parta-launch-20261101-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_depth_parta --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-depth-parta-20261101-a01 --session a2_depth_parta_20261101_a01 --code-commit 1be9ccb33fb146e6e6712aff140e8c3346f0fbdb
