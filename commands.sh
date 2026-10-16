#1 +60+a
#th2-q359-ce7454-evaluation-data-ready-20261016-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
TASK_PY=/mnt/local/conda-py311/envs/ce7454_env/bin/python
TASK_EVAL=/mnt/local/_data/deep-llms_th2/ce7454/evaluation
TASK_MANIFEST_SHA=99ff5ec0bcb187ec7678ae436b0cc0cb7d9349d5431ca1bee8b39b8fc48cc42e
"$TASK_PY" -m unittest discover -s ce7454/src -p test_external_evaluation.py -v
"$TASK_PY" -u ce7454/src/prepare_evaluation_data.py --raw "$TASK_EVAL/raw" --cache "$TASK_EVAL/cache" --manifest-sha256 "$TASK_MANIFEST_SHA"
"$TASK_PY" -u ce7454/src/install_external_evaluation.py --cache "$TASK_EVAL/cache" --manifest-sha256 "$TASK_MANIFEST_SHA"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
