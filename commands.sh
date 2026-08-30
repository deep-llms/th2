#1 +60+a
#th2-verify-nested-phase1-completion-and-start-burns-20260830-a01
set -euo pipefail

TASK_PROJECT_DIR=/mnt/local/@PROJECT@
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/nested_ladder_phase1_20260829"
TASK_WATCHER_SOURCE="$TASK_PROJECT_DIR/scripts/watch_nested_phase1_then_burn_v2.sh"
TASK_WATCHER_RUNTIME=/tmp/watch_nested_phase1_then_burn_v2_71ff6dd7a6f4.sh

echo '=== launch corrected isolated burn watcher v2 ==='
date -u
hostname
cd "$TASK_PROJECT_DIR"
test -x "$TASK_PYTHON"
test -s "$TASK_WATCHER_SOURCE"
echo '71ff6dd7a6f409870365f162a3100319c3c010d01205c493573a531315471665  scripts/watch_nested_phase1_then_burn_v2.sh' \
    | sha256sum -c -

# The canonical watcher remains tracked in the project. Run an exact temporary
# copy so later commands.sh synchronization cannot alter its open script file.
cp "$TASK_WATCHER_SOURCE" "$TASK_WATCHER_RUNTIME"
cmp "$TASK_WATCHER_SOURCE" "$TASK_WATCHER_RUNTIME"
chmod 700 "$TASK_WATCHER_RUNTIME"

export SPARSE_EMB_PROJECT_DIR="$TASK_PROJECT_DIR"
export SPARSE_EMB_PYTHON="$TASK_PYTHON"
export SPARSE_EMB_OUTPUT_BASE="$TASK_OUTPUT_BASE"
export SPARSE_EMB_LOG_DIR="$TASK_LOG_DIR"

exec bash "$TASK_WATCHER_RUNTIME"
