#1 +15+a
#th2-q359-p7-speed-trial-health-20261015-a03
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,time
from scripts.gpu_status import snapshot
from scripts.train_then_burn import burn_progress,approved_launcher,process,BURN_HASH
from scripts.verified_gpu_reclaim import ownership
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-speed-trial-20261015-a01')
print('AT',datetime.now(timezone.utc).isoformat())
for leaf in ('preflight.json','fa4-api.json','supervised/supervisor.json','supervised/run/run.json','supervised/run/timings/run.json','supervised/run/timing-summary.json','supervised/burn-verified.json'):
 p=root/leaf
 if p.exists():print('STATE',leaf,p.read_text())
 else:print('MISSING',leaf)
import statistics
for folder in sorted((root/'supervised/run/timings').glob('*/seed-42/P7-simple-targetgrad')):
 paths=[folder/f'step-profile-rank{rank}.json' for rank in range(8)]
 if all(p.exists() for p in paths):
  ranks=[json.loads(p.read_text()) for p in paths]
  values=[max(r['steps'][i]['interval_seconds'] for r in ranks) for i in range(19,49)]
  result=json.loads((folder/'result.json').read_text())
  print('TIMING',folder.parents[1].name,json.dumps(dict(median=statistics.median(values),intervals=values,peak_gib=max(r['peak_gib'] for r in ranks),evaluation=result['evaluation'])))
for p in sorted((root/'supervised/run').glob('*.log'))+sorted((root/'supervised/run/timings').glob('*.log')):
 print('JOB_TAIL',str(p),p.read_text(errors='replace')[-1800:])
p=root/'supervisor.log'
if p.exists():print('SUPERVISOR_TAIL',p.read_text()[-8000:])
status=snapshot(list(range(8)));print('GPU_SNAPSHOT',json.dumps(status))
b=root/'supervised/burn-verified.json';log=root/'supervised/burn.log'
if b.exists() and log.exists():
 recorded=json.loads(b.read_text());source=str(Path.cwd()/'resources/llm_pretrain_burn.py')
 workers={pid for gpu in status for pid in gpu['pids']}
 parents_ok=all(approved_launcher(process(pid)['ppid'],{source:BURN_HASH}) for pid in workers)
 print('BURN_LIVE',json.dumps(dict(ownership_matches=ownership(status)==ownership(recorded['gpus']),workers=len(workers),approved_parents=parents_ok,progress=burn_progress(log.read_text()),log_age_seconds=time.time()-log.stat().st_mtime)))
 print('BURN_TAIL','\n'.join(log.read_text().splitlines()[-3:]))
print('GUARD_DISABLED',Path('/mnt/local/_gpu_guard/DISABLED').exists())
PY
