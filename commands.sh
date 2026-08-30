#1 +60+a
#th2-launch-corrected-nested-phase1-burn-watcher-v2-20260830-a01
set -euo pipefail

TASK_PROJECT_DIR=/mnt/local/@PROJECT@
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/nested_ladder_phase1_20260829"
TASK_WATCHER_SOURCE="$TASK_PROJECT_DIR/scripts/watch_nested_phase1_then_burn_v2.sh"
TASK_WATCHER_RUNTIME=/tmp/watch_nested_phase1_then_burn_v2_653a1c44595d.sh
TASK_EXPECTED_RUNNER_FRAGMENT="run_experiments.py --experiments 21 22 --stop-at-step 10000 --log-dir $TASK_LOG_DIR"

echo '=== launch corrected isolated burn watcher v2 ==='
date -u
hostname
cd "$TASK_PROJECT_DIR"
test -x "$TASK_PYTHON"
test -s "$TASK_WATCHER_SOURCE"
echo '653a1c44595dd5b26fa3b0dfb9b12bfd1c886c874a7b4a7403b6d8d745fe8161  scripts/watch_nested_phase1_then_burn_v2.sh' \
    | sha256sum -c -

TASK_MATCH_COUNT=0
while read -r TASK_PID; do
    [ -n "$TASK_PID" ] || continue
    TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
    if [[ "$TASK_CMDLINE" == *"$TASK_EXPECTED_RUNNER_FRAGMENT"* ]]; then
        TASK_MATCH_COUNT=$((TASK_MATCH_COUNT + 1))
        echo "verified_active_runner_pid=$TASK_PID cmdline=$TASK_CMDLINE"
    fi
done < <(pgrep -f '[r]un_experiments.py' || true)
test "$TASK_MATCH_COUNT" -eq 1

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
