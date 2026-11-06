#1 +20+a
#th2-wc69-family-mix-prepare-20261107-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=4
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PYCPU'
from pathlib import Path
import json,shutil,hashlib,subprocess,socket,shlex,sys,os
from efficient_llm.a2_family.common import write,sha,SPEC_SHA
if socket.gethostname()!='thiennh-p6-wc69-worker-0':raise ValueError('Wrong node')
source=Path.cwd();root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-mixture-20261107-a01');session='a2_family_mix_20261107_a01'
if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Existing preparation session')
root.mkdir(parents=True,exist_ok=False);frozen=root/'source';files={}
selected=[]
for directory in ('efficient_llm','scripts'):
 selected.extend((source/directory).rglob('*.py'))
selected.extend(source/name for name in ['docs/A2_family_spec_v1.3.md','resources/a2_family_mixture_plan_20261107.json','resources/a2_family_mix_cpu_tests_20261107.json','tests/a2/test_family.py'])
for p in selected:
 rel=p.relative_to(source);q=frozen/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);files[str(rel)]=sha(q)
if sha(frozen/'docs/A2_family_spec_v1.3.md')!=SPEC_SHA:raise ValueError('Wrong spec')
commit='49672971804940346fbcba2e0e0ceb878e3b2abb';write(root/'source-manifest.json',dict(files=files,code_commit=commit))
proof=json.loads((frozen/'resources/a2_family_mix_cpu_tests_20261107.json').read_text())
if proof['status']!='PASS' or any(sha(frozen/p)!=v for p,v in proof['files'].items()):raise ValueError('CPU proof identity changed')
if sha(frozen/'resources/a2_family_mixture_plan_20261107.json')!='12b259426fd63e0f4f27da6b039f9b55a39b33fff052ef55738dea7f6cea15e2':raise ValueError('Mixture plan changed')
cx=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-preparation-20261107-a02/cx')
if sha(cx/'complete.json')!='c163d1d3ec4c9563dde7085911eddd9b904cf413346abe31a453e1efdeab7d27':raise ValueError('CX identity changed')
cx_info=json.loads((cx/'complete.json').read_text())
for split,info in cx_info['sets'].items():
 if sha(cx/split/'documents.json')!=info['documents_sha256']:raise ValueError('CX document metadata changed')
cmd='exec '+shlex.join([sys.executable,'-u','-m','scripts.a2_family_prepare_mix','--plan','resources/a2_family_mixture_plan_20261107.json','--output',str(root/'mixture'),'--cx-root','/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-preparation-20261107-a02/cx'])+' >'+shlex.quote(str(root/'supervisor.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),cmd],check=True)
subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
write(root/'launch.json',dict(root=str(root),session=session,code_commit=commit,cpu_only=True,source_manifest_sha256=sha(root/'source-manifest.json')))
print('FAMILY_CPU_PREPARATION_SUBMITTED',root,session,commit,flush=True)
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
r['mixture_preparation']={str(p.relative_to(mix)):artifact(p) for p in mix.rglob('*.json') if 'source' not in p.relative_to(mix).parts}
r['mixture_log']=tail(mix/'supervisor.log')
r['mixture_session']=subprocess.run(['tmux','display-message','-p','-t','a2_family_mix_20261107_a01','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip()
print('FAMILY_PREPARATION_HEALTH',json.dumps(r),flush=True)
HEALTH
