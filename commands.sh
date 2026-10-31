#1 +20+a
#th2-6y3o-a2-breadth-launch-20261031-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_breadth --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01 --session a2_breadth_20261031_a01 --code-commit 60e5461cbbf21ae437149a45923371cbc6dd74a2
