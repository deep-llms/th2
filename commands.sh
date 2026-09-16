#1 +60+a
#th2-ccm-node-status-after-partial-clear-20260916-a01
set -euo pipefail
date -u
hostname
echo '--- gpu state (read-only)'
/usr/bin/python3 scripts/gpu_status.py
echo '--- gpu compute apps'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true
echo '--- identities of current GPU pids'
for TASK_PID in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
  ps -p "$TASK_PID" -o pid,ppid,user,etime,cmd --no-headers 2>/dev/null | cut -c1-200 || true
done
echo '--- our a2 tmux owner'
tmux has-session -t ccm_scaleup28_phase_a2_20260916_a01 2>/dev/null && echo a2-session-present || echo a2-session-absent
echo CCM_NODE_STATUS_AFTER_PARTIAL_CLEAR_DONE
