#1 +120+a
#th2-watch-pure-local-10k-eval-finetune-then-burn-20260825
set -euo pipefail

echo '=== launch fail-closed Pure-local handoff watcher ==='
date -u
hostname

TASK_PROJECT_DIR="$PWD"
TASK_WATCHER="$TASK_PROJECT_DIR/scripts/watch_pure_local_10k_eval_finetune_burn.sh"
TASK_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
test -s "$TASK_WATCHER"
test -x "$TASK_EVAL_PYTHON"

export SPARSE_EMB_PROJECT_DIR="$TASK_PROJECT_DIR"
export SPARSE_EMB_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
export SPARSE_EMB_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
export SPARSE_EMB_EVAL_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/eval
export SPARSE_EMB_BENCH_ROOT=/mnt/local/_data/@PROJECT@/benchmarks/hf
export SPARSE_EMB_EVAL_PYTHON="$TASK_EVAL_PYTHON"

exec bash "$TASK_WATCHER"
