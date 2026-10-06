#1 +60+a
#th2-tjx3-p4-fa4-status-20261006-a02
set -euo pipefail
cd /mnt/local/@PROJECT@
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PY'
import ast,json,re,socket
from datetime import datetime,timezone
from pathlib import Path
from scripts.gpu_status import snapshot
assert socket.gethostname()=='thiennh-p6-tjx3-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/p4-fa4-four-head-2500-20261006-a01/supervised')
print('OBSERVED_AT',datetime.now(timezone.utc).isoformat(),flush=True)
for name in ('supervisor.json','run/run.json','run/complete.json','burn-verified.json',
             'run/training/P4-iso-4h-validation.json','run/training/P4-4h-validation.json',
             'run/training/seed-42/comparison.json'):
 p=root/name
 if p.is_file():print('FILE',name,p.read_text(),flush=True)
for arm in ('P4-iso-4h','P4-4h'):
 folder=root/'run/training/seed-42'/arm
 log=root/('run/training-seed-42-arm-'+arm+'.log')
 if log.is_file():
  with log.open('rb') as f:
   f.seek(max(0,log.stat().st_size-250000));text=f.read().decode(errors='replace')
  steps=re.findall(r'\|\s*(\d+)/28600',text)
  metrics=[]
  for line in text.splitlines():
   if "'grad_norm':" in line or "'eval_lm_loss':" in line:
    try:
     row=ast.literal_eval(line[line.index('{'):]);metrics.append({k:v for k,v in row.items() if k in ('loss','grad_norm','learning_rate','seconds_per_update','step_lm_loss','step_aux_loss','proxy_lambda','epoch','eval_lm_loss','eval_aux_loss','eval_rows')})
    except (ValueError,SyntaxError):pass
  print('PROGRESS',arm,json.dumps(dict(last_progress_step=int(steps[-1]) if steps else None,
    log_updated_at=datetime.fromtimestamp(log.stat().st_mtime,timezone.utc).isoformat(),metrics=metrics[-4:],
    errors=[line for line in text.splitlines() if any(word in line for word in ('Traceback','RuntimeError','OutOfMemoryError','AssertionError'))][-10:])),flush=True)
  print('LOG_TAIL',arm,text[-3000:],flush=True)
 if folder.is_dir():
  checkpoints=sorted((p for p in folder.glob('checkpoint-*') if p.is_dir()),key=lambda p:int(p.name.split('-')[-1]))
  print('CHECKPOINTS',arm,[p.name for p in checkpoints],flush=True)
  for name in ('result.json','trainer_state.json'):
   p=folder/name
   if p.is_file():
    r=json.loads(p.read_text())
    if name=='trainer_state.json':r={k:r.get(k) for k in ('global_step','max_steps','epoch')}
    print('ARM_FILE',arm,name,json.dumps(r),flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
PY
