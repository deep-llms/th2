#1 +60+a
#th2-tjx3-proxy-speed-validation-20261008-a05
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/proxy-speed-validation-20261008-a05
TASK_SOURCE=/mnt/local/_outputs/@PROJECT@/proxy-speed-validation-20261008-a04/supervised/run
TASK_SESSION=tjx3-proxy-speed-validation-20261008-a05
test ! -e "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
mkdir -p "$TASK_ROOT"
printf -v TASK_CMD 'exec bash %q %q continue %q >%q 2>&1' "$PWD/scripts/launch_proxy_speed_validation.sh" "$TASK_ROOT" "$TASK_SOURCE" "$TASK_ROOT/launch.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -n 80 "$TASK_ROOT/launch.log"
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0
