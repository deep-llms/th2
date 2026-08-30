#1 +180+a
#th2-eval-finetune-four-phase1-models-then-burn-20260830-a03
set -euo pipefail

TASK_PROJECT_DIR=/mnt/local/@PROJECT@
TASK_WORKFLOW_SOURCE="$TASK_PROJECT_DIR/scripts/run_phase1_four_model_eval_finetune_burn.sh"
TASK_WORKFLOW_RUNTIME=/tmp/run_phase1_four_model_eval_finetune_burn_2949bab98ea9.sh

echo '=== launch isolated four-model eval/finetune/burn workflow from free GPUs ==='
date -u
hostname
cd "$TASK_PROJECT_DIR"
test -s "$TASK_WORKFLOW_SOURCE"
echo '2949bab98ea9be829463026d7b2c098e1aba522f7d869280326bcbf4dad45ce6  scripts/run_phase1_four_model_eval_finetune_burn.sh' \
    | sha256sum -c -
cp "$TASK_WORKFLOW_SOURCE" "$TASK_WORKFLOW_RUNTIME"
cmp "$TASK_WORKFLOW_SOURCE" "$TASK_WORKFLOW_RUNTIME"
chmod 700 "$TASK_WORKFLOW_RUNTIME"

export SPARSE_EMB_PROJECT_DIR="$TASK_PROJECT_DIR"
export SPARSE_EMB_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
export SPARSE_EMB_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
export SPARSE_EMB_EVAL_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/eval
export SPARSE_EMB_BENCH_ROOT=/mnt/local/_data/@PROJECT@/benchmarks/hf
export SPARSE_EMB_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
export SPARSE_EMB_CONDA=/mnt/local/conda-py311/bin/conda

exec bash "$TASK_WORKFLOW_RUNTIME"
