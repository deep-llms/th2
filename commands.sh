#1 +300+a
#th2-eval-finetune-tiered-c512-groupreduce-lb-10k-20260905-a01
set -euo pipefail

TASK_PROJECT_DIR=/mnt/local/@PROJECT@
TASK_WORKFLOW="$TASK_PROJECT_DIR/scripts/run_tiered_groupreduce_eval_finetune_burn.sh"
cd "$TASK_PROJECT_DIR"
test -s "$TASK_WORKFLOW"
echo 'f451ab0b43cf46f072bacf27df0b4c03e1963463c7e8a89c085de721b42d963c  scripts/run_tiered_groupreduce_eval_finetune_burn.sh' | sha256sum -c -
bash -n "$TASK_WORKFLOW"

export SPARSE_EMB_PROJECT_DIR="$TASK_PROJECT_DIR"
export SPARSE_EMB_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
export SPARSE_EMB_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
export SPARSE_EMB_EVAL_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/eval
export SPARSE_EMB_BENCH_ROOT=/mnt/local/_data/@PROJECT@/benchmarks/hf
export SPARSE_EMB_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
export SPARSE_EMB_CONDA=/mnt/local/conda-py311/bin/conda

exec bash "$TASK_WORKFLOW"
