#1 +30+a
#th2-6y3o-a2-corpora-cpu-20261029-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_corpora \
 --plan /mnt/local/deep-llms_th2/resources/a2_corpora_30b_20261029.json \
 --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-corpora-20261029-a01 \
 --host thiennh-p6-6y3o-worker-0 --session a2_corpora_20261029_a01 \
 --code-commit fb88f39f19f425bef2e31e9ffb625b2100bfaf6d
