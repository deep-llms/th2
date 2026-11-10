#1 +20+a
#th2-wc69-family-v17-health-20261110-a49
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PYCPU'
from pathlib import Path
import json,datetime,subprocess
from scripts.gpu_process import inspect,process
from scripts.a2_family_v17_pause import digest
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-pause-20261109-a01')
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inspection=inspect(),pause={},pause_log=(root/'pause.log').read_text()[-18000:])
for p in sorted((root/'pause').glob('*.json')):r['pause'][p.name]=dict(path=str(p),sha256=digest(p),value=json.loads(p.read_text()))
for name in ['supervisor.json','burn-verified.json']:
 p=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-20261107-a01/supervised')/name
 if p.exists():r[name]=dict(path=str(p),sha256=digest(p),value=json.loads(p.read_text()))
p=Path('/mnt/local/_gpu_guard/DISABLED');r['guard']=json.loads(p.read_text()) if p.exists() else None
r['chains']={}
for g in r['inspection']['gpus']:
 for pid in g['pids']:
  chain=[]
  while pid>1:
   v=process(pid);chain.append(v);pid=v['ppid']
  r['chains'][str(g['index'])]=chain
admission=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-admission-20261109-a02')
run=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-20261109-a02')
r['admission_log']=(admission/'admission.log').read_text()[-18000:] if (admission/'admission.log').exists() else None
r['v17']={}
for rel in ['progress.json','launch.json','jobs.json','source-manifest.json','supervised/supervisor.json','results/a2_family_20261107/config/plan.json','results/a2_family_20261107/manifests/actual_inputs.json','results/a2_family_20261107/manifests/actual_inputs_at_gpu_admission.json','results/a2_family_20261107/progress.json','results/a2_family_20261107/ledger.json','results/a2_family_20261107/backup/verified.json']:
 p=run/rel
 if p.exists():r['v17'][rel]=dict(path=str(p),sha256=digest(p),value=json.loads(p.read_text()))
if (run/'source-manifest.json').exists():
 m=json.loads((run/'source-manifest.json').read_text());bad=[p for p,h in m['files'].items() if digest(run/'source'/p)!=h]
 r['v17_source']=dict(files=len(m['files']),changed=bad)
r['v17_logs']={p.name:p.read_text()[-6500:] for p in sorted((run/'results/a2_family_20261107/logs').glob('*.log'))}
r['v17_checks']={p.relative_to(run).as_posix():json.loads(p.read_text()) for p in (run/'results/a2_family_20261107/heals').glob('*/tests-rank*.json')}
r['v17_results']={}
for p in (run/'results/a2_family_20261107/heals').glob('*/result.json'):
 v=json.loads(p.read_text());r['v17_results'][p.parent.name]=dict(path=str(p),sha256=digest(p),delta_post=v.get('delta_post'),final=v['scores']['final']['loss'] if v['scores']['final'] else None,registered_run=v['registered_run'],flags=v['flags'])
r['v17_supervisor_log']=(run/'supervisor.log').read_text()[-7000:] if (run/'supervisor.log').exists() else None
r['v17_recipes']={}
for p in (run/'results/a2_family_20261107/heals').glob('*/recipe.json'):
 v=json.loads(p.read_text());r['v17_recipes'][p.parent.name]=dict(path=str(p),sha256=digest(p),value={k:v[k] for k in ('run_id','dataset','arm','teacher','initial_parent','block','removed','steps','tokens','lr','training','source','registered_run','spec_sha256')})
import re
r['v17_nonfinite_counts']={p.name:len(re.findall(r'NONFINITE_KD_FLAG|Traceback \(most recent call last\)|loss[\x27\x22]?\s*:\s*(?:nan|inf)',p.read_text(),flags=re.I)) for p in (run/'results/a2_family_20261107/logs').glob('*.log')}
saved=r['pause']['resume_state.json']['value']
r['paused_checkpoint_members']={}
for side in ['checkpoint','original']:
 base=Path(saved[side]); members={}
 for name in saved['hashes']:
  p=base/name
  members[name]=dict(exists=p.is_file(),bytes=p.stat().st_size if p.is_file() else None)
  if p.is_file() and name.endswith('.json'):members[name]['sha256']=digest(p)
 r['paused_checkpoint_members'][side]=members
r['input_file_sizes']=[dict(path=x['path'],expected=x['bytes'],actual=Path(x['path']).stat().st_size if Path(x['path']).is_file() else None) for x in r['v17']['results/a2_family_20261107/manifests/actual_inputs.json']['value']['files']]

r['handoff_artifacts']={}
for p in sorted((run/'supervised').rglob('*.json')):
 if p.stat().st_size<2000000:r['handoff_artifacts'][str(p.relative_to(run))]=dict(sha256=digest(p),value=json.loads(p.read_text()))
for name in ['decision_v17.json','tests.json']:
 p=run/'results/a2_family_20261107'/name
 if p.exists():r['handoff_artifacts'][name]=dict(sha256=digest(p),value=json.loads(p.read_text()))
r['worker_commands']={}
for pid in r['inspection']['workers']:
 p=Path('/proc')/str(pid)
 if p.exists():r['worker_commands'][str(pid)]=dict(argv=(p/'cmdline').read_bytes().decode().split('\0'),cwd=str((p/'cwd').resolve()))

print('V17_HEALTH '+json.dumps(r),flush=True)
PYCPU

/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'HEALTH'
from pathlib import Path
import json,hashlib,subprocess
from datetime import datetime,timezone
from scripts.gpu_process import inspect,process
from scripts.a2_remaining import read_locked
from scripts.train_then_burn import digest
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-depth-partc-20261103-a01');base=root/'results/a2_depth_20261031'
def artifact(p):
 if not p.is_file():return None
 raw=p.read_bytes();v=json.loads(raw) if p.suffix=='.json' else raw.decode(errors='replace')
 return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),value=v)
def tail(p,n=4500):
 if not p.is_file():return None
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-n));v=f.read().decode(errors='replace')
 return dict(path=str(p),bytes=p.stat().st_size,text=v)
r=dict(utc=datetime.now(timezone.utc).isoformat(),inspection=inspect());chains={}
for worker in r['inspection']['workers']:
 pid=worker;rows=[]
 while pid>1 and len(rows)<12:
  try:v=process(pid)
  except (FileNotFoundError,ProcessLookupError):break
  rows.append(v);pid=v['ppid']
 chains[str(worker)]=rows
r['chains']=chains;r['continuations']=artifact(root/'continuations.json')
latest=r['continuations']['value']['attempts'][-1];attempt=Path(latest['root'])
r['native']={p:artifact(attempt/p) for p in ['admission.json','inspection.json','supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/run.json','supervised/run/complete.json','supervised/run/depth-partc-complete.json','supervised/burn-verified.json']}
r['source_checks']=[]
for folder,manifest in [(root/'source',json.loads((root/'source-manifest.json').read_text())),(attempt/'source',read_locked(latest['execution_source_manifest']))]:
 changed=[p for p,sha in manifest['files'].items() if digest(folder/p)!=sha]
 r['source_checks'].append(dict(path=str(folder),files=len(manifest['files']),changed=changed))
r['execution_manifest']=artifact(attempt/'execution-source-manifest.json')
r['progress']={p:artifact(base/p) for p in ['progress.json','DT6_checks.json','DT6.json','coordinator_checks.json','attempts.json','decision_depth_partc.json','downstream/progress.json','downstream/decision_downstream.json']}
r['arms']={}
for arm in ['X72','P300','P1T']:
 folder=base/'heals'/f'ST1-dclm-{arm}-S1-12.8B-lr1e-3';tests=[]
 for p in sorted(folder.glob('tests-rank*.json')):
  a=artifact(p);c=a['value']['checks'];tests.append(dict(path=str(p),sha256=a['sha256'],checks=len(c),flags=[x for x in c if not x['passed']],resume=[x for x in c if x['test']=='resume_available']))
 result=artifact(folder/'result.json')
 if result:
  v=result['value'];result['value']={k:v.get(k) for k in ['status','run_id','flags','unverified','delta_post','roundtrip_difference','resume_events']};result['value']['scores']={k:None if x is None else x.get('loss') for k,x in v['scores'].items()};result['value']['updates']=len(v['lr_updates'])
 checkpoints=[]
 for p in sorted(folder.glob('trainer/**/depth-checkpoint.json')):
  try:
   marker=json.loads(p.read_text());actual={name:(p.parent/name).stat().st_size for name in marker['files']}
   checkpoints.append(dict(path=str(p),step=marker['global_step'],complete=marker['complete'],finite=marker.get('state_finite'),actual_sizes_match=actual==marker['files']))
  except FileNotFoundError as error:checkpoints.append(dict(path=str(p),observation='member disappeared during snapshot; possible native rotation',error=repr(error)))
 r['arms'][arm]=dict(tests=tests,resume=artifact(folder/'resume_events.json'),result=result,checkpoints=checkpoints)
r['logs']=[tail(p) for p in [attempt/'supervisor.log',attempt/'supervised/run/a2-depth-partc.log',attempt/'supervised/burn.log',*sorted(base.glob('logs/C-*.log'))]]
r['nonfinite_flags']={}
for p in base.glob('logs/C-*.log'):
 with p.open(errors='replace') as f:r['nonfinite_flags'][p.name]=sum('DEPTH_NONFINITE_KD_FLAG' in line or 'DEPTH_NONFINITE_RECOVERY' in line for line in f)
guard=Path('/mnt/local/_gpu_guard/DISABLED');r['guard']=guard.read_text() if guard.exists() else None
r['session']=subprocess.run(['tmux','display-message','-p','-t',latest['session'],'#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip()
r['cpu_processes']=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  cmd=(proc/'cmdline').read_bytes().split(b'\0')
  if b'scripts.a2_depth_partc_run' not in cmd:continue
  item=process(int(proc.name));item.update(cwd=str((proc/'cwd').resolve()),io=(proc/'io').read_text(),status=(proc/'status').read_text().splitlines()[:8]);r['cpu_processes'].append(item)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
print('DEPTH_RESUME_HEALTH',json.dumps(r),flush=True)
HEALTH
