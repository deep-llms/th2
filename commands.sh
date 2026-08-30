#1 +180+a
#th2-eval-finetune-four-phase1-models-then-burn-20260830-a01
set -euo pipefail

TASK_PROJECT_DIR=/mnt/local/@PROJECT@
TASK_WORKFLOW_SOURCE="$TASK_PROJECT_DIR/scripts/run_phase1_four_model_eval_finetune_burn.sh"
TASK_WORKFLOW_RUNTIME=/tmp/run_phase1_four_model_eval_finetune_burn_6a8d9c6825a9.sh

echo '=== launch isolated four-model eval/finetune/burn workflow ==='
date -u
hostname
cd "$TASK_PROJECT_DIR"
test -s "$TASK_WORKFLOW_SOURCE"
echo '6a8d9c6825a92c908b4c7032fe43ce4708580ba87706f29002ee279f94150235  scripts/run_phase1_four_model_eval_finetune_burn.sh' \
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
