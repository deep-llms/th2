#1 +60+a
#th2-readonly-raw-tiered-unified-training-progress-20260906-a02
set -euo pipefail
date -u
hostname
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
ps -eo pid,ppid,etime,comm | grep -E 'python|accelerate' || true
python3 - <<'PY'
from pathlib import Path
import json
base = Path('/mnt/local/_outputs/@PROJECT@')
logs = base/'logs/raw_tiered_unified_ranklift_10k_20260906_a01'
def tail(p, size=14000):
    print('\nFILE', p, flush=True)
    if not p.is_file():
        print('NOT PRESENT'); return
    with p.open('rb') as f:
        f.seek(max(0, p.stat().st_size-size))
        print(f.read().decode(errors='replace').replace('\r','\n'))
tail(logs/'experiments.log', 6000)
for arm in ('tiered_ranklift_raw_t4_c512','unified_ranklift_raw_t4_m460'):
    tail(logs/(arm+'.log'))
    checkpoints = sorted((base/arm).glob('checkpoint-*'), key=lambda p:int(p.name.split('-')[-1]))
    print('CHECKPOINTS', arm, [p.name for p in checkpoints])
    if checkpoints:
        state = checkpoints[-1]/'trainer_state.json'
        if state.is_file():
            s=json.loads(state.read_text())
            print('TRAINER_STATE', arm, 'global_step', s.get('global_step'), 'latest_metrics', s.get('log_history', [])[-3:])
marker=base/'status/raw_tiered_unified_ranklift_10k_20260906_a01.complete'
print('COMPLETION_MARKER', marker.exists())
if marker.exists(): print(marker.read_text())
tail(base/'logs/gpu_burn_after_raw_tiered_unified_ranklift_20260906_a01.log', 2500)
PY
echo 'TH2 READONLY RAW TIERED UNIFIED PROGRESS CHECK COMPLETE'
