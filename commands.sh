#1 +30+a
#th2-tjx3-p6-questions-readonly-20261008-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - <<'PY'
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,re
from safetensors import safe_open
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-remaining-2500-20261006-a01/supervised/run/training/seed-42/P6-iso')
state=root/'trainer_state.json';checkpoint=root/'checkpoint-2500/model.safetensors'
print('UTC',datetime.now(timezone.utc).isoformat(),flush=True)
for p in (state,checkpoint):print('SOURCE',str(p),p.is_file(),p.stat().st_size if p.is_file() else None,flush=True)
if state.is_file():
 raw=state.read_bytes();x=json.loads(raw);print('STATE_SHA256',hashlib.sha256(raw).hexdigest(),'STEP',x['global_step'],flush=True)
 series=[]
 for row in x['log_history']:
  d={k:float(v) for k,v in row.items() if re.fullmatch(r'proxy_layer_\d+_mean_abs_alpha_0',k)}
  if d and (row['step']==0 or row['step']%250==0):series.append({'step':row['step'],'alpha':d})
 print('ALPHA_TRAJECTORY',json.dumps(series),flush=True)
if checkpoint.is_file():
 with safe_open(checkpoint,framework='pt',device='cpu') as f:
  alpha={k:float(f.get_tensor(k).float().abs().mean()) for k in f.keys() if re.fullmatch(r'heads\.\d+\.alpha',k)}
 print('ALPHA_CHECKPOINT',json.dumps(alpha),flush=True)
follow=Path('/mnt/local/_outputs/deep-llms_th2/proxy-followup-2500-20261008-a01/supervised/run')
if (follow/'run.json').is_file():
 q=json.loads((follow/'run.json').read_text())
 print('FOLLOWUP_QUEUE',json.dumps({'status':q['status'],'jobs':[{'name':j['name'],'status':j['status'],'started_at':j.get('started_at'),'finished_at':j.get('finished_at')} for j in q['jobs']]},flush=True))
 for seed in (42,1042):
  for arm in ('P7-simple-sparse','P7-simple-short','A','P6-iso','P7-simple'):
   p=follow/f'seed-{seed}'/arm/'result.json'
   if p.is_file():
    x=json.loads(p.read_text());print('RESULT',str(p),x['global_step'],x['evaluation'].get('eval_lm_loss'),flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
print('DONE',flush=True)
PY
