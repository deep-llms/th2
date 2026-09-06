#1 +120+a
#th2-restore-runner-burn-after-ranklift-completion-20260906-a01
set -euo pipefail
date -u
hostname
TASK_BASE=/mnt/local/_outputs/@PROJECT@
test -s "$TASK_BASE/status/raw_tiered_unified_ranklift_10k_20260906_a01.complete"
grep -Fx 'STATUS=SUCCESS' "$TASK_BASE/status/raw_tiered_unified_ranklift_10k_20260906_a01.complete"
test -s /tmp/llm_pretrain_burn.py
echo '=== existing tmux sessions ==='
tmux list-sessions || true
echo '=== old burn log tail ==='
tail -n 12 "$TASK_BASE/logs/gpu_burn_after_raw_tiered_unified_ranklift_20260906_a01.log"
if pgrep -af '[t]rain_compositional.py|[r]un_experiments.py|[e]val_parallel.py|[f]inetune_parallel.py'; then
    echo 'Refusing burn: project workload still exists'; exit 1
fi
for attempt in 1 2; do
    TASK_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
    test -z "$TASK_GPU_PIDS" || { echo 'GPU processes exist; refusing launch'; exit 1; }
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
    sleep 3
done
TASK_SESSION=ranklift_completed_burn_20260906_a01
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then
    echo 'Refusing duplicate burn session'; exit 1
fi
tmux new-session -d -s "$TASK_SESSION" "exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/bin/python3 -u /tmp/llm_pretrain_burn.py >> '$TASK_BASE/logs/gpu_burn_ranklift_restore_20260906_a01.log' 2>&1"
sleep 40
python3 - <<'PY'
import subprocess
from pathlib import Path
seen=set()
for gpu in range(8):
    pids=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).split()
    assert len(pids)==1, (gpu,pids)
    pid=pids[0]; assert pid not in seen; seen.add(pid)
    current=pid; owned=False
    for depth in range(6):
        if int(current)<=1: break
        p=Path('/proc')/current
        cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if '/tmp/llm_pretrain_burn.py' in cmd:
            owned=True; print('BURN_GPU',gpu,'worker',pid,'ancestor',current,cmd); break
        current=next(l.split()[1] for l in (p/'status').read_text().splitlines() if l.startswith('PPid:'))
    assert owned,(gpu,pid)
print('ALL_EIGHT_BURN_WORKERS_VERIFIED')
PY
tmux has-session -t "$TASK_SESSION"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
tail -n 12 "$TASK_BASE/logs/gpu_burn_ranklift_restore_20260906_a01.log"
echo 'TH2 RUNNER BURN RESTORED IN INDEPENDENT TMUX SESSION'
