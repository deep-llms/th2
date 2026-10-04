#1 +60+a
#th2-tjx3-document-training-status-20261004-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python -u - <<'PY'
import json,statistics
from pathlib import Path
from scripts.gpu_status import snapshot
from run_experiments import now
root=Path('/mnt/local/_outputs/deep-llms_th2/document-training-20261004-a01')
print('READ_ONLY_STATUS',now(),flush=True)
print('GPU_STATUS',json.dumps(snapshot()),flush=True)
for mode in ['sdpa_cross','sdpa_causal','fa4_cross','sdpa_isolated','fa4_isolated']:
    paths=[root/'benchmark'/mode/f'rank-{i}.json' for i in range(8)]
    if all(p.exists() for p in paths):
        ranks=[json.loads(p.read_text()) for p in paths]
        times=[max(r['steps'][i]['interval_seconds'] for r in ranks) for i in range(5,30)]
        print('COMPLETED_MODE',json.dumps(dict(mode=mode,steps=len(ranks[0]['steps']),median_seconds=statistics.median(times),peak_gib=max(r['peak_allocated_gib'] for r in ranks))),flush=True)
    else:
        p=root/(mode+'.log')
        print('INCOMPLETE_MODE',mode,'log_exists',p.exists(),flush=True)
        if p.exists():print(p.read_text()[-1500:],flush=True)
if (root/'result.json').exists():print('FINAL_RESULT',(root/'result.json').read_text(),flush=True)
PY
