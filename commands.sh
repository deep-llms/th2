#1 +60+a
#th2-ohmv-embedding-anchor-qualification-20261122-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH="$PWD:$PWD/tests/warm_order"
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" -m unittest discover -s tests/warm_order -p 'test_embedding_anchor*.py'
"$TASK_PYTHON" -m scripts.launch_embedding_anchor_qualification --commit 182c1e1e99e73c5fc7149426f2f8d1b5eb077c1f --source-sha e0949ee4ec2230909ffb13a094e9a04f43f296ad0c5ab7b0f1f783dc22c9aa93 --plan-sha 2d7554927d9adf33e4d63a94d697b9075719456ac80e359a014fdb8e55617eaa
