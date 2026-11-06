#1 +20+a
#th2-wc69-depth-resume-health-20261106-a08
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
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
pf=read_locked(dict(path='/mnt/local/_outputs/deep-llms_th2/wc69-a2-depth-resume-preflight-20261106-a01.json',sha256='bbb9c3d0dc28b8637fad183d852ce75cc7c1cba06753faee2fa3186fb1575aca'))
read_locked(pf['pause_receipt']);cfg=read_locked(pf['depth_launch']['config']);stream=cfg['heal_streams']['dclm']
read_locked(dict(path=stream['path']+'/manifest.json',sha256=stream['sha256']))
r['input_locks']=dict(pause_receipt_unchanged=True,config_unchanged=True,heal_manifest_unchanged=True,last_actual_data_hash_verification=pf['utc'],last_actual_checkpoint_22sha_verification=pf['utc'],full_data_rehashed_this_health=False)
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
