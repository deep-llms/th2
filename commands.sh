#1 +60+a
#th2-78gg-performance-status-20260929-a04
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-78gg-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python - <<'PY'
import json,statistics
from pathlib import Path
from datetime import datetime,timezone
root=Path('/mnt/local/_outputs/deep-llms_th2/performance-20260929-a01')
print('PERF_STATUS',datetime.now(timezone.utc).isoformat(),flush=True)
for name in ['preflight.json','production/gpus-free-before-training.json','production/supervisor.json','production/run/run.json','production/run/benchmark-summary.json','production/burn-verified.json']:
 path=root/name
 if path.exists():print('ARTIFACT',name,path.read_text(),flush=True)
 else:print('ABSENT',name,flush=True)
for case in json.loads((root/'variants.json').read_text()):
 name=case[0];path=root/'production/run'/name
 if all((path/f'benchmark-rank{i}.json').exists() for i in range(8)):
  ranks=[json.loads((path/f'benchmark-rank{i}.json').read_text()) for i in range(8)]
  wall=[max(r['steps'][i]['interval_seconds'] for r in ranks) for i in range(5,18)]
  compute=[max(r['steps'][i]['compute_seconds'] for r in ranks) for i in range(5,18)]
  print('MEASUREMENT',json.dumps(dict(name=name,seconds=statistics.mean(wall),compute_seconds=statistics.mean(compute),
   stdev=statistics.stdev(wall),peak_gib=max(r['allocated_peak_bytes'] for r in ranks)/2**30,
   validation=ranks[0]['validation'],wrapped_model=ranks[0]['wrapped_model'])),flush=True)
 log=root/'production/run'/f'{name}.log'
 if log.exists():
  with log.open('rb') as f:
   f.seek(max(0,log.stat().st_size-7000));text=f.read().decode(errors='replace')
  print('LOG_TAIL',name,'\n'+'\n'.join(text.replace('\r','\n').splitlines()[-12:]),flush=True)
PY
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu --format=csv
