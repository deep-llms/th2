#1 +30+a
#th2-6y3o-dot-pilot-status-20261020-a10
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
python3 - <<'PY'
from pathlib import Path
import datetime,json
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-pilot-20261020-a01')
print('OBSERVED_UTC',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
for name in ['preparation-failed.json','assets-verified.json','cache-ready.json','launch.json','prepared/budget.json','prepared/validation.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/run.json','supervised/run/T12/summary.json','supervised/run/T14/initial.json','supervised/run/T14/summary.json','supervised/supervisor.json','supervised/burn-verified.json','supervised/run/D.verified.json','supervised/run/P1-A.verified.json','supervised/run/P1-B.verified.json']:
 p=root/name
 if not p.exists():continue
 j=json.loads(p.read_text())
 if name=='assets-verified.json':j={k:j[k] for k in ['status','files_verified','splits']}
 if name=='cache-ready.json':j={k:j[k] for k in ['status','seconds','train_sequences','validation_sequences']}
 if name=='prepared/validation.json':j={**{k:v for k,v in j.items() if k not in ['identity','periodic_indices','final_indices','overlapping_validation_documents','dropped_sequences']},'periodic_sequences':len(j['periodic_indices']),'final_sequences':len(j['final_indices']),'overlapping_documents':len(j['overlapping_validation_documents']),'dropped_sequences':len(j['dropped_sequences'])}
 if name.endswith('T12/summary.json'):j={k:v for k,v in j.items() if k!='results'}
 print('RECEIPT',name,json.dumps(j),flush=True)
for p in [Path(str(root)+'.prepare.log'),root/'supervisor.log',root/'supervised/run/T12.log',root/'supervised/run/T14.log',root/'supervised/run/D.log',root/'supervised/run/P1-A.log',root/'supervised/run/P1-B.log']:
 if not p.exists():continue
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2200));s=f.read().decode(errors='replace')
 print('LOG_TAIL',str(p),'\n'+s,flush=True)
for prefix in ['supervised/run/T12','supervised/run/T14']:
 path=root/prefix
 logs=sorted(path.glob('*.log'),key=lambda p:p.stat().st_mtime) if path.exists() else []
 if logs:
  p=logs[-1]
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-1600));s=f.read().decode(errors='replace')
  print('CHECK_LOG',p.name,'\n'+s,flush=True)
for arm in ['D','P1-A','P1-B']:
 path=root/'prepared/runs'/arm
 if not path.exists():continue
 print('ARM',arm,'CHECKPOINTS',sorted(p.name for p in path.glob('checkpoint-*')),flush=True)
 for name in ['metrics.jsonl','train_results.json','final_eval.json']:
  p=path/name
  if not p.exists():continue
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3000));s=f.read().decode(errors='replace')
  print('ARM_RESULT',arm,name,'\n'+s,flush=True)
from scripts.gpu_process import inspect
print('GPU_LIVE',json.dumps(inspect()),flush=True)

import hashlib,re,time
for arm in ['D','P1-A','P1-B']:
 path=root/'prepared/runs'/arm
 if not (path/'train_results.json').exists():continue
 final=path/'checkpoint-4768'
 names=['optimizer.pt','scheduler.pt','trainer_state.json']+[f'rng_state_{i}.pth' for i in range(8)]
 missing=[n for n in names if not (final/n).is_file() or (final/n).stat().st_size==0]
 weights=list(final.glob('*.safetensors'))
 state=json.loads((final/'trainer_state.json').read_text())
 print('FINAL_CHECK',arm,json.dumps(dict(step=state['global_step'],max_steps=state['max_steps'],missing=missing,weight_files=len(weights),checkpoints=len(list(path.glob('checkpoint-*'))))),flush=True)
 for name in ['train_results.json','final_eval.json','metrics.jsonl']:
  p=path/name
  if p.exists():print('RESULT_SHA256',arm,name,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
supervisor=root/'supervised/supervisor.json'
state=json.loads(supervisor.read_text()) if supervisor.exists() else {}
if state.get('finished_at') and (root/'supervised/burn-verified.json').exists():
 from scripts.native_baseline_pipeline import burn_gate
 from scripts.train_then_burn import burn_progress
 burn_gate(root)
 p=root/'supervised/burn.log'
 before=p.read_text()
 records=re.findall(r'gpu_burn_progress rank=0 completed_cycles=(\d+) .*?completed_collective_payload_gib=([\d.]+)',before)
 time.sleep(5)
 after=p.read_text()
 current=re.findall(r'gpu_burn_progress rank=0 completed_cycles=(\d+) .*?completed_collective_payload_gib=([\d.]+)',after)
 print('LIVE_BURN_PROGRESS',json.dumps(dict(all_rank_readiness_and_progress=burn_progress(after),before=records[-2:],after=current[-2:])),flush=True)
 burn_gate(root)
PY
