#1 +30+a
#th2-6y3o-attention-compare-status-20261020-a04
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
python3 - <<'CHECK'
import json,datetime,subprocess
from pathlib import Path
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-attention-compare-20261020-a01')
print('OBSERVED_UTC',datetime.datetime.now(datetime.timezone.utc).isoformat())
for name in ('supervisor.log','supervised/supervisor.json','supervised/run/attention_comparison.log','supervised/burn-verified.json'):
 p=root/name
 if p.is_file():print('FILE',name,p.read_text()[-14000:],flush=True)
checks=root/'supervised/run/checks'
for name in ('progress.json','selection.json'):
 p=checks/name
 if not p.exists():continue
 x=json.loads(p.read_text());rows=x if isinstance(x,list) else x['results']
 if isinstance(x,dict):print('SELECTION',json.dumps({k:v for k,v in x.items() if k!='results'}))
 for r in rows:
  print('ENV_RESULT',json.dumps({k:r.get(k) for k in ('name','eligible','reason','benchmarks')}))
  for c in r['checks']:
   print('NUMERICS',json.dumps({k:c.get(k) for k in ('name','passed','returncode','seconds')}))
   if not c['passed']:
    log=checks/(c['name']+'.log')
    if log.exists():print('FAILED_CHECK_LOG',log.read_text()[-5000:])
if (root/'supervised/burn-verified.json').exists():
 from scripts.gpu_process import inspect, ownership
 proof=json.loads((root/'supervised/burn-verified.json').read_text());live=inspect()
 print('LIVE_BURN_HANDOFF',json.dumps(dict(matches=ownership(proof['gpus'])==ownership(live['gpus']),guard_disabled=live['guard_disabled'],workers=live['workers'])))
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv'],text=True))
CHECK
