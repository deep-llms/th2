#1 +240+a
#th2-train-ranklift-tied-c124-m460-10k-20260902-a01
set -euo pipefail

TASK_PROJECT_DIR=/mnt/local/@PROJECT@
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_DATA_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/train
TASK_CONDA=/mnt/local/conda-py311/bin/conda
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11

cd "$TASK_PROJECT_DIR"
eval "$("$TASK_CONDA" shell.bash hook)"
conda activate sparse_emb

test "$(command -v python3.11)" = "$TASK_PYTHON"
test "$CONDA_DEFAULT_ENV" = sparse_emb

exec "$TASK_PYTHON" -u scripts/run_ranklift_b200_10k.py \
    --project-dir "$TASK_PROJECT_DIR" \
    --output-base "$TASK_OUTPUT_BASE" \
    --model-dir "$TASK_MODEL_DIR" \
    --data-dir "$TASK_DATA_DIR" \
    --python "$TASK_PYTHON"
