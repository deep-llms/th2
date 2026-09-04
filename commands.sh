#1 +300+a
#th2-run-five-checkpoint-eval-finetune-burn-20260904-a01
set -euo pipefail

TASK_PROJECT_DIR=/mnt/local/@PROJECT@
TASK_WORKFLOW_SOURCE="$TASK_PROJECT_DIR/scripts/run_five_checkpoint_eval_finetune_burn.sh"
TASK_WORKFLOW_COPY=/tmp/run_five_checkpoint_eval_finetune_burn.sh
TASK_EXPECTED_SHA256=d0b7c485f89e2cf8a1965128c04423f4e266e2df10f11b445574ee99dff30682

cd "$TASK_PROJECT_DIR"
test -s "$TASK_WORKFLOW_SOURCE"
TASK_ACTUAL_SHA256="$(sha256sum "$TASK_WORKFLOW_SOURCE" | awk '{print $1}')"
test "$TASK_ACTUAL_SHA256" = "$TASK_EXPECTED_SHA256"
cp "$TASK_WORKFLOW_SOURCE" "$TASK_WORKFLOW_COPY"
chmod 0700 "$TASK_WORKFLOW_COPY"
test "$(sha256sum "$TASK_WORKFLOW_COPY" | awk '{print $1}')" = "$TASK_EXPECTED_SHA256"

export SPARSE_EMB_PROJECT_DIR="$TASK_PROJECT_DIR"
export SPARSE_EMB_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
export SPARSE_EMB_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
export SPARSE_EMB_EVAL_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/eval
export SPARSE_EMB_BENCH_ROOT=/mnt/local/_data/@PROJECT@/benchmarks/hf
export SPARSE_EMB_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
export SPARSE_EMB_CONDA=/mnt/local/conda-py311/bin/conda

exec bash "$TASK_WORKFLOW_COPY"
