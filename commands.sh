#1 +60+a
#th2-wc69-family-preparation-health-20261107-a84
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

prep=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-preparation-20261107-a02')
r['preparation']={str(p.relative_to(prep)):artifact(p) for p in prep.rglob('*.json') if 'source' not in p.relative_to(prep).parts}
r['preparation_logs']=[tail(p) for p in prep.glob('*.log')]
r['preparation_session']=subprocess.run(['tmux','display-message','-p','-t','a2_family_prepare_20261107_a02','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip()
r['family_model_files']=[dict(path=str(p),bytes=p.stat().st_size) for p in Path('/mnt/local/_models/deep-llms_th2/a2_family').rglob('*') if p.is_file()]
mix=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-mixture-20261107-a01')
r['mixture_preparation']={str(p.relative_to(mix)):artifact(p) for p in [mix/'launch.json',mix/'mixture/progress.json',mix/'mixture/complete.json',*mix.glob('mixture/sources/*/progress.json')] if p.is_file()}
r['mixture_log']=tail(mix/'supervisor.log')
r['mixture_session']=subprocess.run(['tmux','display-message','-p','-t','a2_family_mix_20261107_a01','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip()

repair=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-mixture-20261107-a02')
r['mixture_repair']={str(p.relative_to(repair)):artifact(p) for p in [repair/'launch.json',repair/'mixture/progress.json',repair/'mixture/complete.json',*repair.glob('mixture/sources/*/progress.json')] if p.is_file()}
r['mixture_repair_log']=tail(repair/'supervisor.log')
r['mixture_repair_session']=subprocess.run(['tmux','display-message','-p','-t','a2_family_mix_20261107_a02','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip()
r['code_content_files']=[]
fix=json.loads(Path('resources/a2_family_mixture_content_correction_20261107.json').read_text())
for group,rows in fix['download_groups'].items():
 for entry in rows:
  p=Path(entry['local']);item=dict(path=str(p),exists=p.is_file(),expected_size=entry['bytes'],expected_sha256=entry['sha256'])
  if p.is_file():item.update(bytes=p.stat().st_size,sha256=digest(p));item['verified']=item['bytes']==entry['bytes'] and item['sha256']==entry['sha256']
  item['parent_members']=[dict(name=q.name,bytes=q.stat().st_size,is_file=q.is_file()) for q in p.parent.iterdir()] if p.parent.is_dir() else []
  r['code_content_files'].append(item)


admit=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-admission-20261107-a01')
r['family_admission']={str(p.relative_to(admit)):artifact(p) for p in [admit/'launch.json',admit/'admission/progress.json',admit/'admission/failure.json',admit/'admission/complete.json',admit/'admission/pause/request.json',admit/'admission/pause/saved_state.json',admit/'admission/pause/terminal.json'] if p.is_file()}
r['family_admission_log']=tail(admit/'supervisor.log')
r['family_admission_session']=subprocess.run(['tmux','display-message','-p','-t','a2_family_admission_20261107_a01','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip()
if (admit/'source-manifest.json').is_file():
 m=json.loads((admit/'source-manifest.json').read_text());r['family_admission_source']=dict(code_commit=m['code_commit'],files=len(m['files']),changed=[p for p,h in m['files'].items() if digest(admit/'source'/p)!=h])
family=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-20261107-a01')
r['family_native']={str(p.relative_to(family)):artifact(p) for p in [family/'launch.json',family/'supervised/supervisor.json',family/'supervised/run/run.json',family/'results/a2_family_20261107/progress.json'] if p.is_file()}


r['family_logs']=[tail(p,6500) for p in [family/'supervisor.log',family/'supervised/run/a2-family.log',*sorted((family/'results/a2_family_20261107/logs').glob('*.log'))] if p.is_file()]
r['family_checks']={str(p.relative_to(family)):artifact(p) for p in (family/'results/a2_family_20261107').glob('**/tests-rank*.json')}
r['family_other']={str(p.relative_to(family)):artifact(p) for p in [family/'results/a2_family_20261107/preparation_checks.json',family/'results/a2_family_20261107/ledger.json',family/'results/a2_family_20261107/lr_selection.json',family/'results/a2_family_20261107/coordinator_checks.json',family/'supervised/reclaim.json',family/'supervised/gpus-free-before-training.json'] if p.is_file()}


launch=read_locked(dict(path=str(family/'launch.json'),sha256=digest(family/'launch.json')))
cfg=read_locked(launch['config']);fm=read_locked(launch['source_manifest'])
r['family_source_verification']=dict(code_commit=fm['code_commit'],files=len(fm['files']),changed=[p for p,h in fm['files'].items() if digest(family/'source'/p)!=h],manifest=launch['source_manifest'])
r['family_config']=artifact(Path(launch['config']['path']))
paused=read_locked(cfg['depth_pause']);saved=read_locked(paused['saved_state']);r['family_pause_saved']=saved
r['family_mixture_manifest']=artifact(Path(cfg['mixture_stream']['path'])/'manifest.json')
r['family_input_lock_checks']={}
for name in ['preparation','models_manifest','cx_complete','mixture_complete','mixture_plan','depth_config','depth_execution_manifest','overlap_preflight']:
 read_locked(cfg[name]);r['family_input_lock_checks'][name]=True
mixplan=read_locked(cfg['mixture_plan']);t=mixplan['tokenizer'];r['shared_tokenizer_check']=dict(path=t['local'],expected=t['sha256'],actual=digest(t['local']))
def compact(value):
 if isinstance(value,dict):return {k:compact(v) for k,v in value.items() if not k.startswith('per_sequence') and k!='lr_updates'}
 if isinstance(value,list):return [compact(v) for v in value]
 return value
r['family_result_summaries']={}
for p in (family/'results/a2_family_20261107').glob('**/result.json'):
 r['family_result_summaries'][str(p.relative_to(family))]=dict(path=str(p),sha256=digest(p),value=compact(json.loads(p.read_text())))
r['family_nonfinite_log_counts']={}
for p in (family/'results/a2_family_20261107/logs').glob('*.log'):
 with p.open(errors='replace') as f:r['family_nonfinite_log_counts'][p.name]=sum('FAMILY_NONFINITE_KD_FLAG' in line for line in f)

r['family_jobs']=read_locked(launch['jobs'])
r['family_completion_backup']={str(p.relative_to(family)):artifact(p) for p in [family/'results/a2_family_20261107/backup/verified.json',family/'results/a2_family_20261107/decision_family.json',family/'supervised/run/complete.json',family/'supervised/run/family-complete.json'] if p.is_file()}

transition=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v14-transition-20261108-a01')
r['family_v14_transition']={p.name:artifact(p) for p in [transition/'installed.json',transition/'ready.json',transition/'gate-status.json',transition/'attempt.json',transition/'f6-complete.json'] if p.is_file()}
if (transition/'installed.json').is_file():
 installed=json.loads((transition/'installed.json').read_text())
 r['family_source_verification']['authorized_bridge_sha256']=installed['patched_sha256']
 r['family_source_verification']['authorized_bridge_verified']=digest(family/'source/scripts/a2_family_depth_bridge.py')==installed['patched_sha256']
 r['family_source_verification']['unexpected_changes']=[p for p,h in fm['files'].items() if digest(family/'source'/p)!=(installed['patched_sha256'] if p=='scripts/a2_family_depth_bridge.py' else h)]
 r['family_v14_gate_verified']=digest(transition/'gate.py')==installed['gate_sha256']
f6=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-20261108-a02')
r['family_f6']={str(p.relative_to(f6)):artifact(p) for p in [f6/'progress.json',f6/'source-manifest.json',f6/'results/a2_family_20261107/config/plan.json',f6/'results/a2_family_20261107/manifests/actual_inputs.json',f6/'results/a2_family_20261107/progress.json',f6/'results/a2_family_20261107/extension_decision.json',f6/'results/a2_family_20261107/backup/verified.json'] if p.is_file()}
r['family_f6_logs']=[tail(p,6500) for p in sorted((f6/'results/a2_family_20261107/logs').glob('*.log'))]
for f6source in [f6,Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-preparation-20261108-a03')]:
 if (f6source/'source-manifest.json').is_file():
  sm=json.loads((f6source/'source-manifest.json').read_text())
  r.setdefault('family_f6_source',{})[str(f6source)]=dict(files=len(sm['files']),code_commit=sm['code_commit'],changed=[n for n,h in sm['files'].items() if digest(f6source/'source'/n)!=h])
r['family_f6_processes']=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\0')
  if any(x in args for x in [b'scripts.a2_family_f6_prepare',b'scripts.a2_family_f6_run',b'efficient_llm.a2_family_f6.heal']):r['family_f6_processes'].append(process(int(p.name)))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass

r['family_f6_checks']={str(p.relative_to(f6)):artifact(p) for p in (f6/'results/a2_family_20261107').glob('**/tests-rank*.json')}
r['family_f6_results']={str(p.relative_to(f6)):dict(path=str(p),sha256=digest(p),value=compact(json.loads(p.read_text()))) for p in (f6/'results/a2_family_20261107').glob('**/result.json')}
r['family_f6_ledger']=artifact(f6/'results/a2_family_20261107/ledger.json')
r['bridge_log']=tail(family/'supervised/run/a2-family-depth-bridge.log',12000)
r['family_f6_coordinator']=artifact(f6/'results/a2_family_20261107/coordinator_checks.json')
r['family_f6_nonfinite']={}
for p in (f6/'results/a2_family_20261107/logs').glob('*.log'):
 with p.open(errors='replace') as f:r['family_f6_nonfinite'][p.name]=sum('NONFINITE' in line or 'Traceback (most recent call last)' in line for line in f)
r['family_depth_admission']=artifact(family/'supervised/run/depth-resume-admission.json')
r['family_depth_bridge_log']=tail(family/'supervised/run/family_depth_bridge.log',12000)
r['snapshot_complete_utc']=datetime.now(timezone.utc).isoformat()
print('FAMILY_PREPARATION_HEALTH',json.dumps(r),flush=True)
HEALTH
