#1 +240+a
#th2-eval-finetune-final-rse-hashed-10k-then-correct-burn-20260901-a01
set -euo pipefail

TASK_PROJECT_DIR=/mnt/local/@PROJECT@
TASK_WORKFLOW_SOURCE="$TASK_PROJECT_DIR/scripts/run_final_rse_hashed_eval_finetune_burn.sh"
TASK_WORKFLOW_RUNTIME=/tmp/run_final_rse_hashed_eval_finetune_burn_f954755720e1.sh
TASK_WORKFLOW_SHA=f954755720e1d65f797627bbce7c365bfba3960438a4f8937ab43941e1124b4f

echo '=== launch final RSE + Hashed eval/finetune/burn workflow ==='
date -u
hostname
cd "$TASK_PROJECT_DIR"
test -x "$TASK_WORKFLOW_SOURCE"
echo "$TASK_WORKFLOW_SHA  $TASK_WORKFLOW_SOURCE" | sha256sum -c -
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
