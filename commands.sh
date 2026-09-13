#1 +30+a
#th2-ccm-readonly-burn-check-20260913-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-8mgy-worker-0
/usr/bin/python3 scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7
/usr/bin/python3 scripts/ccm_smoke_gpu_control.py verify-burn
tmux display-message -p -t ccm_burn_20260913_a01 '#{session_name} pane_dead=#{pane_dead} pane_pid=#{pane_pid}'
sleep 10
/usr/bin/python3 scripts/ccm_smoke_gpu_control.py verify-burn
date -u
echo CCM_READONLY_ALL_EIGHT_BURNS_VERIFIED
