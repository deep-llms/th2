#1 +30+a
#th2-pq5r-hold-lifetime-guard-inspect-20261128-a01
set -euo pipefail
printf 'READ_ONLY_HOLD_AND_GUARD_INSPECTION (no process is signaled)\n'
date -u
id
printf 'PID1_START %s  ELAPSED_S %s\n' "$(ps -o lstart= -p 1)" "$(ps -o etimes= -p 1)"
printf 'PID432 %s  ELAPSED_S %s\n' "$(ps -o lstart= -p 432 2>/dev/null || echo gone)" "$(ps -o etimes= -p 432 2>/dev/null || echo gone)"
if test -r /proc/1/environ; then
  tr '\0' '\n' </proc/1/environ | grep -E '^RUN_HOURS=' || printf 'RUN_HOURS unset in PID1 environ (script default 8)\n'
else
  printf 'PID1 environ not readable\n'
fi
uptime
printf '=== gpu_guard.sh ===\n'
cat /mnt/local/_gpu_guard/gpu_guard.sh
printf '=== burn.pids ===\n'
cat /mnt/local/_gpu_guard/burn.pids 2>&1 | head -20
ls -la /mnt/local/_gpu_guard/
ps -eo pid,ppid,lstart,args | grep -E "gpu_guard|polite_burn" | grep -v grep | cut -c1-200 || true
printf 'INSPECTION_DONE\n'
