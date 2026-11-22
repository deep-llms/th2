#1
#th2-ohmv-anchor-boundary-cpu-20261122-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1
prior=/mnt/local/_outputs/deep-llms_th2/embedding_anchor_fit_20261122_a01
test "$(hostname)" = thiennh-p6-ohmv-worker-0
study="$PWD/scripts/anchor_boundary_study.py"
cd "$prior/execution/source"
export PYTHONPATH="$PWD"
exec /mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$study" --out /mnt/local/_outputs/deep-llms_th2/anchor_boundary_cpu_20261122_a01
