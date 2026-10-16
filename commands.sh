#1 +60+a
#th2-q359-ce7454-interim-eval-smoke-20261016-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_PY=/mnt/local/conda-py311/envs/ce7454_env/bin/python
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-ce7454-interim-smoke-20261016-a01
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$TASK_PY" -m unittest discover -s ce7454/src -p test_interim_evaluation.py -v
test ! -e "$TASK_ROOT"
mkdir -p "$TASK_ROOT/code"
cp ce7454/src/*.py "$TASK_ROOT/code/"
cp ce7454/source_manifest.json "$TASK_ROOT/source_manifest.json"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv
"$TASK_PY" -u "$TASK_ROOT/code/interim_watch.py" --output "$TASK_ROOT/probe" --once --arms grip_nodown band_strong
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv
