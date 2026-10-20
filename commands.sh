#1 +30+a
#th2-6y3o-native12-final-status-20261020-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
python3 - <<'REMOTE_PY'
from pathlib import Path
import datetime,json,re,time
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-native12-cx-en-20261020-a02')
print('OBSERVED_UTC',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
for name in ['supervised/run/run.json','supervised/supervisor.json','supervised/burn-verified.json','supervised/run/baseline-5000.json','prepared/baseline/train_results.json']:
 p=root/name
 if p.is_file():
  j=json.loads(p.read_text())
  if name.endswith('/run.json'):j={**{k:v for k,v in j.items() if k!='jobs'},'jobs':[{k:v for k,v in x.items() if k!='artifacts'} for x in j['jobs']]}
  if name.endswith('baseline-5000.json'):j={k:v for k,v in j.items() if k!='files'}
  print('RECEIPT',name,json.dumps(j),flush=True)
 else:print('MISSING',name,flush=True)
base=root/'prepared/baseline'
print('CHECKPOINTS',sorted(int(p.name.split('-')[-1]) for p in base.glob('checkpoint-*')),flush=True)
if (base/'train_results.json').is_file():
 from scripts.training_results import check_training
 verified=check_training(base,5000,57220)
 print('FINAL_ARTIFACT_CHECK',json.dumps({k:v for k,v in verified.items() if k!='files'}),flush=True)
 state=json.loads((base/'trainer_state.json').read_text())
 print('VALIDATION_HISTORY',json.dumps([x for x in state['log_history'] if 'eval_loss' in x]),flush=True)
from scripts.gpu_process import inspect,process
from scripts.native_baseline_pipeline import validate_burn_handoff
from scripts.train_then_burn import burn_progress,approved_launcher,BURN_HASH
live=inspect()
print('GPU_LIVE',json.dumps(live),flush=True)
proof=root/'supervised/burn-verified.json'
if proof.is_file():
 state=json.loads((root/'supervised/supervisor.json').read_text());proof=json.loads(proof.read_text())
 validate_burn_handoff(state,proof,live)
 assert all(approved_launcher(process(pid)['ppid'],{str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}) for pid in live['workers'])
 log=root/'supervised/burn.log'
 first=log.read_text();time.sleep(10);second=log.read_text()
 pattern=r'gpu_burn_progress rank=0 completed_cycles=(\d+) .*?completed_collective_payload_gib=([\d.]+)'
 a=re.findall(pattern,first)[-1];b=re.findall(pattern,second)[-1]
 assert burn_progress(second) and int(b[0])>int(a[0]) and float(b[1])>float(a[1])
 print('LIVE_BURN_COLLECTIVES_VERIFIED',json.dumps({'before':a,'after':b,'guard_released':not live['guard_disabled']}),flush=True)
for name in ['supervisor.log','supervised/run/baseline-5000.log']:
 p=root/name
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-8192));text=f.read().decode(errors='replace')
  print('LOG_TAIL',name,'\n'+'\n'.join(text.splitlines()[-12:]),flush=True)
REMOTE_PY
