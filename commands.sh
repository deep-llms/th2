#1 +30+a
#th2-6y3o-a2-stage1-shuffle-20261029-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_a2_stage1_data --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1-shuffle-20261029-a01 --data-output /mnt/local/_data/deep-llms_th2/a2_stage1_dclm_shuffle2030_20261029 --session a2_stage1_shuffle_20261029_a01 --code-commit 716d3f3521f4f34823dc32fc7ebca43ae3049f73
