#1 +120+a
#th2-train-dense-tied-ddp-default-10k-then-burn-20260829-a01
set -euo pipefail

export SPARSE_EMB_PROJECT_DIR="$PWD"
export SPARSE_EMB_CONDA=/mnt/local/conda-py311/bin/conda
export SPARSE_EMB_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
export SPARSE_EMB_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
export SPARSE_EMB_DATA_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/train
export SPARSE_EMB_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@

exec bash "$PWD/scripts/run_dense_ddp_default_then_burn.sh"
