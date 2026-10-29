#1 +30+a
#th2-6y3o-a2-stage1b-fwe-shuffle-20261030-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_stage1b_data --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1b-fwe-shuffle-20261030-a01 --data-output /mnt/local/_data/deep-llms_th2/a2_stage1b_fwe_shuffle2030_20261030 --session a2_stage1b_fwe_shuffle_20261030_a01 --code-commit f5e56ecf1776a2b566e83e8488989eb2b7d645e7
