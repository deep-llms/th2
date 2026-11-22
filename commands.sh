#1 +60+a
#th2-ohmv-embedding-anchor-fit-20261122-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH="$PWD:$PWD/tests/warm_order"
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" -m unittest test_embedding_anchor_fit
"$TASK_PYTHON" -m scripts.launch_embedding_anchor_fit --commit 36614fc9c0d70bb41fe90732672577e3b086df8c --source-sha 74c1504c3a891d74bdb805070968cf6228e409d8863946b5bcf9453b78f775e8 --plan-sha 6d72d3acd59076ed195d67267b3a1aaa79851d56cd7b1c6ee4470ef1762537f0
