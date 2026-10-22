#1 +30+a
#th2-6y3o-dot-C1-final-status-20261022-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'PY'
import hashlib,io,json,re,tarfile
from pathlib import Path
from scripts.gpu_process import inspect,process
from scripts.native_baseline_pipeline import validate_burn_handoff
from scripts.train_then_burn import burn_progress,approved_launcher,BURN_HASH
from scripts.bdot_job import latest_complete_checkpoint
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-C1-20261022-a01')
sup=root/'supervised';run=sup/'run';train=root/'prepared/runs/C1-s0'
read=lambda p:json.loads(p.read_text())
live=inspect();state={'live':live,'queue':read(run/'run.json'),'supervisor':read(sup/'supervisor.json')}
files=[]
for name in ['supervised/run/run.json','supervised/supervisor.json','supervised/run/complete.json',
 'supervised/run/C1-summary.json','supervised/run/C1-s0.verified.json','supervised/run/C1-s0/result.json',
 'supervised/run/C1-s0/routes_sample.npz','C1-contract.json','prepared/C1-s0.train.json','prepared/C1-s0.model.json',
 'prepared/runs/C1-s0/train_results.json','prepared/runs/C1-s0/final_eval.json','prepared/runs/C1-s0/metrics.jsonl',
 'prepared/runs/C1-s0/retention.jsonl']:
 p=root/name
 if p.is_file():
  files.append(p)
  if p.suffix=='.json':state[name]=read(p)
state['retained_checkpoints']=[p.name for p in sorted(train.glob('checkpoint-*'))]
if list(train.glob('checkpoint-*')):
 cp=latest_complete_checkpoint(train);s=read(cp/'trainer_state.json')
 state['checkpoint']={'path':str(cp),'global_step':s['global_step'],'max_steps':s['max_steps'],
  'files':{p.name:p.stat().st_size for p in cp.iterdir() if p.is_file()}}
 for name in ['trainer_state.json','config.json','train_config.json']:
  files.append(cp/name)
if (sup/'burn-verified.json').is_file():
 proof=read(sup/'burn-verified.json');files.append(sup/'burn-verified.json');state['burn_receipt']=proof
 try:
  validate_burn_handoff(state['supervisor'],proof,live)
  source=str(Path.cwd()/'resources/llm_pretrain_burn.py')
  assert all(approved_launcher(process(pid)['ppid'],{source:BURN_HASH}) for pid in live['workers'])
  text=(sup/'burn.log').read_text();assert burn_progress(text)
  state['burn_handoff_live_verified']=True
  state['last_burn_progress']=re.findall(r'^.*gpu_burn_progress.*$',text,re.M)[-3:]
 except Exception as e:state['burn_verification_error']=repr(e)
for j in state['queue'].get('jobs',[]):
 if j.get('status')=='failed':
  for p in [run/(j['name']+'.log'),*run.glob(j['name']+'.verified.attempt*.log')]:
   if p.is_file():
    files.append(p)
    print('FAILED_LOG',p.name,p.read_text(errors='replace')[-16000:])
print('C1_FINAL_STATUS',json.dumps(state),flush=True)
status=root/'final-status-20261022-a01.json'
with status.open('x') as f:json.dump(state,f,indent=2)
files.append(status)
content={str(p.relative_to(root)):p.read_bytes() for p in sorted(set(files))}
archive=root/'final-results-20261022-a01.tar.gz'
if archive.exists():raise RuntimeError('Refusing existing status archive')
with tarfile.open(archive,'w:gz') as t:
 for name,data in content.items():
  info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;t.addfile(info,io.BytesIO(data))
manifest={'archive':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
 'members':{k:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for k,v in content.items()}}
with (root/'final-results-20261022-a01.manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
print('C1_RESULTS_READY',manifest['sha256'],len(content),archive.stat().st_size,flush=True)
PY
