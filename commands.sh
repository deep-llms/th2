#1 +120+a
#th2-ohmv-balance-units-health-20261119-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
import json,time
from pathlib import Path
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect,process
base=Path('/mnt/local/_outputs/deep-llms_th2/balance_units_20261119')
r=dict(utc=time.time(),ownership=inspect(),state={},runs={})
for f in ('status.json','terminal.json','scientific-complete.json','failure.json','backup.json','REPORT.json','execution/admission.json','execution/submitted.json'):
 p=base/f
 if p.exists():r['state'][f]=json.loads(p.read_text())
if (base/'execution/source.json').exists():
 pin=json.loads((base/'execution/source.json').read_text());r['source_actual']={p:sha(base/'execution/source'/p)==v for p,v in pin.items()}
plan=json.loads(Path('resources/balance_units_plan_20261119.json').read_text());hashes=dict(plan['hashes'])
prepared=json.loads((base/'inputs/complete.json').read_text())
for rel,want in prepared['files'].items():hashes[str(base/'inputs'/rel)]=want
r['input_actual']={p:sha(p)==v for p,v in hashes.items()}
guard=Path('/mnt/local/_gpu_guard/DISABLED');r['guard_owner']=json.loads(guard.read_text()) if guard.exists() else None
if r['guard_owner']:
 owner=r['guard_owner']['owner_pid'];r['guard_owner_identity']=process(owner)
 r['guard_owner_parent']=process(r['guard_owner_identity']['ppid'])
r['active_identity']=[]
for job in r['state'].get('status.json',{}).get('active',[]):
 pid=job['pid'];record={'name':job['name'],'pid':pid}
 try:
  actual=[x.decode() for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
  record.update(argv=actual,argv_matches=actual==job['argv'],identity=process(pid))
  parent=record['identity']['ppid'];record['parent']=process(parent) if parent>1 else None
 except FileNotFoundError:record['exited_during_snapshot']=True
 r['active_identity'].append(record)
from efficient_llm.warm_order.balance_units_report import NAMES
for name in NAMES:
 run=base/'runs'/name;q={}
 for f in ('production-probe.json','train/progress.json','train/checkpoint.json','train/resume.json','train/complete.json','complete.json'):
  p=run/f
  if p.exists():q[f]=json.loads(p.read_text())
 trace=run/'train/trace.jsonl'
 if trace.exists():q['train_trace_tail']=[json.loads(x) for x in trace.read_text().splitlines()[-2:]]
 log=base/f'fit_eval-{name}.log'
 if log.exists():q['log_tail']=log.read_text()[-2000:]
 r['runs'][name]=q
r['checkpoint_actual']={}
for name in NAMES:
 run=base/'runs'/name/'train'
 if (run/'complete.json').exists():
  from efficient_llm.warm_order.recursion_eval import check_train
  check_train(run);r['checkpoint_actual'][name]=True
if (base/'native-verified.json').exists():r['native']=json.loads((base/'native-verified.json').read_text())
if (base/'execution/queue.log').exists():r['queue_tail']=(base/'execution/queue.log').read_text()[-3000:]
if r['state'].get('terminal.json',{}).get('success'):
 from scripts.gpu_process import ownership
 from scripts.train_then_burn import approved_launcher,GUARD_HASH
 state=r['ownership'];prior=r['native']['samples'][-1]
 assert not guard.exists() and sha(guard.parent/'gpu_guard.sh')==GUARD_HASH
 assert ownership(state['gpus'])==ownership(prior['gpus'])
 assert all(process(int(pid))==p for pid,p in prior['processes'].items())
 native='/tmp/llm_pretrain_burn.py';assert sha(native)==r['native']['script_sha256']
 assert len(state['workers'])==8 and all(approved_launcher(process(pid)['ppid'],{native:r['native']['script_sha256']}) for pid in state['workers'])
 b=r['state']['backup.json'];assert sha(base/'results.tar.gz')==b['archive_sha256'] and (base/'results.tar.gz').stat().st_size==b['bytes']
 assert all(sha(p)==v for p,v in b['node_copies'].items())
 assert all(sha(base/p)==v for p,v in b['files'].items())
 assert all(r['source_actual'].values()) and all(r['input_actual'].values())
 r['terminal_actual_proof']=dict(native_ownership=True,guard_released=True,all_source_inputs_members_backups=True,backup_sha256=sha(base/'backup.json'))
if r['state'].get('terminal.json',{}).get('native_verified') and not r['state']['terminal.json']['success']:
 from scripts.gpu_process import ownership
 from scripts.train_then_burn import approved_launcher,GUARD_HASH
 state=r['ownership'];prior=r['native']['samples'][-1];native='/tmp/llm_pretrain_burn.py'
 assert not guard.exists() and sha(guard.parent/'gpu_guard.sh')==GUARD_HASH
 assert ownership(state['gpus'])==ownership(prior['gpus'])
 assert all(process(int(pid))==p for pid,p in prior['processes'].items())
 assert sha(native)==r['native']['script_sha256'] and len(state['workers'])==8
 assert all(approved_launcher(process(pid)['ppid'],{native:r['native']['script_sha256']}) for pid in state['workers'])
 r['failure_handoff_proof']=dict(native_ownership=True,guard_released=True)
import numpy as np
r['interim_metrics']={}
for name in NAMES:
 for domain in ('dclm','fwe'):
  p=base/'runs'/name/domain/'summary.npz'
  if p.exists() and (base/'runs'/name/'complete.json').exists():
   a=np.load(p);v=dict(nll=float(a['nll'].mean(dtype=np.float64)),sequences=len(a['nll']))
   if 'routes' in a:v['window_route_histogram']=np.bincount(a['routes'].flatten(),minlength=4).tolist()
   r['interim_metrics'][name+'/'+domain]=v
print('BALANCE_UNITS_HEALTH',json.dumps(r),flush=True)
PY
