#1 +180+a
#th2-eval-finetune-raw-tiered-unified-10k-20260906-a02
set -euo pipefail
TASK_PROJECT_DIR=/mnt/local/@PROJECT@
cd "$TASK_PROJECT_DIR"
echo '27bed3ae0449c4e3ef0ff7dda6f7660c3cc34a0d62ded172983f6d955aed071d  scripts/run_raw_tiered_unified_eval_finetune_burn.sh' | sha256sum -c -
tmux list-panes -a -F '#{session_name} pid=#{pane_pid} dead=#{pane_dead} exit=#{pane_dead_status}' || true
bash -n scripts/run_raw_tiered_unified_eval_finetune_burn.sh
export SPARSE_EMB_PROJECT_DIR="$TASK_PROJECT_DIR"
export SPARSE_EMB_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
export SPARSE_EMB_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
export SPARSE_EMB_EVAL_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/eval
export SPARSE_EMB_BENCH_ROOT=/mnt/local/_data/@PROJECT@/benchmarks/hf
export SPARSE_EMB_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
export SPARSE_EMB_CONDA=/mnt/local/conda-py311/bin/conda
exec bash scripts/run_raw_tiered_unified_eval_finetune_burn.sh
