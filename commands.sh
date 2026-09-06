#1 +60+a
#th2-readonly-verify-raw-tiered-unified-eval-start-20260906-a01
set -euo pipefail
date -u
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
tmux list-panes -a -F '#{session_name} pid=#{pane_pid} dead=#{pane_dead} exit=#{pane_dead_status} signal=#{pane_dead_signal}'
python3 - <<'PY'
from pathlib import Path
base=Path('/mnt/local/_outputs/@PROJECT@')
def tail(p,n=5000):
    print('FILE',p)
    if not p.is_file(): print('NOT PRESENT');return
    with p.open('rb') as f:
        head=f.read(18000).decode(errors='replace')
        f.seek(max(0,p.stat().st_size-n))
        end=f.read().decode(errors='replace')
    for line in head.splitlines():
        if 'Loaded compositional model:' in line: print(line)
    print(end.replace('\r','\n'))
for arm in ('tiered_ranklift_raw_t4_c512','unified_ranklift_raw_t4_m460'):
    tail(base/arm/'checkpoint-10000/eval.log')
tail(base/'logs/eval_finetune_raw_tiered_unified_10k_20260906_a02.log')
tail(base/'logs/burn_eval_raw_tiered_unified_10k_20260906_a02.log',1600)
PY
echo 'TH2 LIVE EVAL READONLY VERIFICATION COMPLETE'
