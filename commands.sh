#1 +60+a
#th2-readonly-ranklift-progress-and-matched-ppl-20260906-a03
set -euo pipefail
date -u
hostname
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
python3 - <<'PY'
from pathlib import Path
import json
base=Path('/mnt/local/_outputs/@PROJECT@')
def tail(path, size=9000):
    print('FILE',path)
    if not path.is_file():
        print('NOT PRESENT'); return
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-size))
        print(f.read().decode(errors='replace').replace('\r','\n'))
logs=base/'logs/raw_tiered_unified_ranklift_10k_20260906_a01'
tail(logs/'experiments.log')
for arm in ('tiered_ranklift_raw_t4_c512','unified_ranklift_raw_t4_m460'):
    tail(logs/(arm+'.log'))
    checkpoints=sorted((base/arm).glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]),reverse=True)
    for checkpoint in checkpoints:
        path=checkpoint/'trainer_state.json'
        if not path.is_file(): continue
        try:
            state=json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        rows=[row for row in state.get('log_history',[]) if 'loss' in row]
        print('EARLY_HISTORY '+json.dumps(dict(arm=arm,checkpoint=checkpoint.name,global_step=state.get('global_step'),rows=rows)))
        break
    else:
        print('NO_READABLE_STATE',arm)
marker=base/'status/raw_tiered_unified_ranklift_10k_20260906_a01.complete'
print('COMPLETION_MARKER',marker.exists())
if marker.exists(): print(marker.read_text())
tail(base/'logs/gpu_burn_after_raw_tiered_unified_ranklift_20260906_a01.log',4000)
PY
echo 'TH2 READONLY MATCHED EARLY PPL COMPLETE'
