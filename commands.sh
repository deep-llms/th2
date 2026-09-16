#1 +60+a
#th2-ccm-node-status-inspect-20260916-a01
set -euo pipefail
date -u
hostname
echo '--- gpu state (read-only)'
/usr/bin/python3 scripts/gpu_status.py
echo '--- gpu process details'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true
echo '--- process identities of current GPU pids'
for TASK_PID in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
  ps -p "$TASK_PID" -o pid,ppid,user,etime,rss,cmd --no-headers 2>/dev/null | cut -c1-220 || true
done
echo '--- memory now'
free -g || true
echo '--- tmux sessions with live panes'
tmux ls 2>/dev/null || echo no-tmux
for TASK_S in $(tmux ls -F '#{session_name}' 2>/dev/null); do
  echo "$TASK_S pane_dead=$(tmux display-message -p -t "$TASK_S" '#{pane_dead}' 2>/dev/null || echo query-failed)"
done
echo '--- our phase-a outputs still present'
ls /mnt/local/_outputs/deep-llms_th2/ccm_scaleup28_phase_a_20260915_a01/ 2>/dev/null | head -20 || echo missing
ls /mnt/local/_data/deep-llms_th2/ccm/scaleup28_v3_20260915_a01/ >/dev/null 2>&1 && echo partial-data-root-present || echo partial-data-root-missing
echo '--- historical outputs intact'
ls -d /mnt/local/_outputs/deep-llms_th2/ccm_pilot_seed17_20260913_a01 /mnt/local/_outputs/deep-llms_th2/ccm_pilot_seed29_20260913_a01 /mnt/local/_outputs/deep-llms_th2/ccm_replication_seed29_20260914_a01 /mnt/local/_data/deep-llms_th2/ccm/pilot_v1_20260913_a01 2>/dev/null || echo some-missing
echo '--- disk free'
df -h /mnt/local | tail -1
echo CCM_NODE_STATUS_INSPECT_DONE
