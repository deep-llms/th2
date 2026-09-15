#1 +60+a
#th2-ccm-scaleup28-failure-inspect-20260915-a01
set -euo pipefail
date -u
hostname
echo '--- kernel OOM evidence'
(dmesg -T 2>/dev/null || dmesg 2>/dev/null || echo dmesg-unavailable) | grep -iE 'killed process|out of memory|oom[-_ ]kill' | tail -30 || echo no-oom-lines
echo '--- memory now'
free -g || true
echo '--- cgroup limits'
cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo no-cgroup-file
cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo no-cgroup-usage
cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo no-peak
echo '--- memory cgroup events'
cat /sys/fs/cgroup/memory.events 2>/dev/null || echo no-events
echo '--- preparation process 97083 state'
ps -p 97083 -o pid,stat,etime,rss,cmd 2>/dev/null || echo prep-process-gone
echo '--- tmux sessions'
tmux ls || echo no-tmux
echo '--- gpu state (read-only)'
/usr/bin/python3 scripts/gpu_status.py
echo '--- failed shallow29 outputs (read-only listing)'
ls -la /mnt/local/_outputs/deep-llms_th2/ccm_scaleup28_phase_a_20260915_a01/shallow12_seed29/train/shallow/ 2>/dev/null || echo missing
echo '--- partial extension data root (read-only listing)'
ls -la /mnt/local/_data/deep-llms_th2/ccm/scaleup28_v3_20260915_a01/ 2>/dev/null || echo missing
echo '--- disk free'
df -h /mnt/local | tail -1
echo CCM_SCALEUP28_FAILURE_INSPECT_DONE
