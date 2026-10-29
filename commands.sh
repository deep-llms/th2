#1 +30+a
#th2-6y3o-a2-stage1-heals-health-20261029-a21
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import hashlib,json,socket,subprocess
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1-heals-20261029-a01')
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
rels=['source-manifest.json','inputs.json','jobs.json','accelerate-env.log','supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/burn-verified.json','supervised/run/run.json','supervised/run/complete.json']
fitroot=root/'supervised/run/results/a2_stage1'
for name in ('progress.json','lr_choice.json','decision_stage1.json','complete.json'):rels.append(str((fitroot/name).relative_to(root)))
for p in fitroot.glob('ST1-*/result.json'):rels.append(str(p.relative_to(root)))
for rel in rels:
 p=root/rel
 if p.is_file():
  raw=p.read_bytes();assert len(raw)<5000000
  print('ARTIFACT',json.dumps(dict(path=rel,sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
 else:print('MISSING',rel,flush=True)
base=root/'supervised/run'
logs=[root/'supervisor.log',root/'supervised/burn.log',*base.glob('*.log'),*fitroot.glob('*.log'),*fitroot.glob('*/profile.txt')]
for p in logs:
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6000));tail=f.read().decode(errors='replace')
  print('LOG_TAIL',json.dumps(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,text=tail)),flush=True)
guard=Path('/mnt/local/_gpu_guard/DISABLED');print('GUARD',guard.read_text() if guard.is_file() else 'released',flush=True)
print('SESSION',subprocess.run(['tmux','display-message','-p','-t','a2_stage1_heals_20261029_a01','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip(),flush=True)
processes=subprocess.run(['ps','-eo','pid,ppid,pcpu,pmem,etime,args'],capture_output=True,text=True,check=True).stdout.splitlines()
print('LAUNCH_PROCESSES',json.dumps([line for line in processes if 'scripts.launch_a2_stage1' in line or str(root/'source') in line]),flush=True)
corpora=Path('/mnt/local/_data/deep-llms_th2/a2_corpora_30b_20261029')
cpu=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-corpora-20261029-a03')
for rel,path in [('corpora/source-manifest.json',cpu/'source-manifest.json'),('corpora/status.json',corpora/'status.json'),('corpora/failure.json',corpora/'failure.json'),('corpora/complete.json',corpora/'complete.json'),('corpora/fineweb_edu/manifest.json',corpora/'packed/fineweb_edu/manifest.json')]:
 if path.is_file():
  raw=path.read_bytes();print('ARTIFACT',json.dumps(dict(path=rel,sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
plan=json.loads((cpu/'source/plan.json').read_text());records=plan['corpora']['fineweb_edu']['files']
files=[Path(f['local'])/Path(f['path']).name if Path(f['local']).is_dir() else Path(f['local']) for f in records]
parts=[]
for p in sorted((corpora/'parts/fineweb_edu').glob('*/complete.json')):
 item=json.loads(p.read_text());assert item['source']==records[item['source']['index']]
 parts.append(dict(index=item['source']['index'],tokens=sum(x['tokens'] for x in item['streams'].values()),seconds=item['seconds'],sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print('FWE_PROGRESS',json.dumps(dict(raw_files=sum(p.is_file() for p in files),raw_bytes=sum(p.stat().st_size for p in files if p.is_file()),parts=parts,observed_tokens=sum(p.stat().st_size for p in (corpora/'parts/fineweb_edu').glob('*/*.bin') if p.name!='documents.bin')//4)),flush=True)
print('FWE_SESSION',subprocess.run(['tmux','display-message','-p','-t','a2_corpora_20261029_a03','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip(),flush=True)
print('HEALTH_COMPLETE',flush=True)
PY
