#1 +120+a
#th2-watch-sampling-verify-stop-burn-launch-dense-baseline-20260828-a01
set -euo pipefail

export SPARSE_EMB_PROJECT_DIR="$PWD"
export SPARSE_EMB_CONDA=/mnt/local/conda-py311/bin/conda
export SPARSE_EMB_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
export SPARSE_EMB_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
export SPARSE_EMB_DATASET_ROOT=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B
export SPARSE_EMB_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
export SPARSE_EMB_BASELINE_LOG_DIR=/mnt/local/_outputs/@PROJECT@/logs/dense_tied_baseline_b200_fresh_b200_20260828

exec bash "$PWD/scripts/watch_sampling_then_train_dense_baseline.sh"
