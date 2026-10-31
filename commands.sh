#1 +20+a
#th2-6y3o-a2-breadth-health-20261031-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import hashlib,json,socket,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect,process
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01');base=root/'results/a2_breadth_20261031'
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
print('HOST',socket.gethostname(),flush=True)
live=inspect();print('GPU_INSPECTION',json.dumps(live),flush=True)
chains={}
for worker in live['workers']:
 pid=worker;chain=[]
 while pid>1 and len(chain)<12:
  try:row=process(pid)
  except (FileNotFoundError,ProcessLookupError):break
  chain.append(row);pid=row['ppid']
 chains[str(worker)]=chain
print('PROCESS_CHAINS',json.dumps(chains),flush=True)
def export(p):
 if p.is_file():
  raw=p.read_bytes()
  if len(raw)<2000000:print('ARTIFACT',json.dumps(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
  else:print('LARGE_ARTIFACT',json.dumps(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))),flush=True)
 else:print('MISSING',str(p.relative_to(root)),flush=True)
for rel in ('source-manifest.json','jobs.json','budget.json','admission.json','accelerate-env.log','supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/burn-verified.json','supervised/run/run.json','supervised/run/complete.json','supervised/run/breadth-complete.json'):
 export(root/rel)
for rel in ('config/plan.json','preparation_checks.json','coordinator_checks.json','progress.json','ledger.json','manifests/documents.json','decision_breadth.json','independent_replay.json','REPORT.md'):
 export(base/rel)
manifest=root/'source-manifest.json'
if manifest.exists():
 m=json.loads(manifest.read_text());changed=[name for name,value in m['files'].items() if hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()!=value]
 print('SOURCE_VERIFICATION',json.dumps(dict(files=len(m['files']),code_commit=m['code_commit'],changed=changed)),flush=True)
for path in sorted(base.glob('heals/*/tests-rank0.json')):export(path)
for path in sorted(base.glob('heals/*/result.json')):export(path)
for path in sorted(base.glob('smol/*/complete.json')):export(path)
for path in sorted(base.glob('perseq/*/*.json')):
 v=json.loads(path.read_text());print('PERSEQ',json.dumps(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),loss=v.get('loss'),sequences=v.get('sequences'),tokens=v.get('tokens'))),flush=True)
plan=base/'config/plan.json'
if plan.exists():
 cfg=json.loads(plan.read_text());ready={}
 for e in cfg['downloads']['checkpoints']:
  ready[e['revision']]={f['name']:dict(exists=(Path(e['path'])/f['name']).exists(),bytes=(Path(e['path'])/f['name']).stat().st_size if (Path(e['path'])/f['name']).exists() else 0,expected=f['bytes']) for f in e['files']}
 print('DOWNLOAD_FILES',json.dumps(ready),flush=True)
for p in [root/'supervisor.log',root/'supervised/burn.log',root/'supervised/run/a2-breadth.log',*base.glob('logs/*.log')]:
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2500));tail=f.read().decode(errors='replace')
  print('LOG_TAIL',json.dumps(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,text=tail)),flush=True)
guard=Path('/mnt/local/_gpu_guard/DISABLED');print('GUARD',guard.read_text() if guard.is_file() else 'released',flush=True)
print('SESSION',subprocess.run(['tmux','display-message','-p','-t','a2_breadth_20261031_a01','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip(),flush=True)
rows=subprocess.run(['ps','-eo','pid,ppid,pcpu,pmem,etime,args'],capture_output=True,text=True,check=True).stdout.splitlines()
print('LAUNCH_PROCESSES',json.dumps([r for r in rows if 'a2_breadth' in r or str(root) in r]),flush=True)
terminal=root/'supervised/supervisor.json'
if terminal.exists() and json.loads(terminal.read_text()).get('finished_at'):
 samples=[]
 for i in range(2):
  snap=inspect();proc={}
  for pid in snap['workers']:
   item=process(pid);proc[str(pid)]=item
   if item['ppid']>1:proc[str(item['ppid'])]=process(item['ppid'])
  log=root/'supervised/burn.log';samples.append(dict(inspection=snap,processes=proc,burn_tail=log.read_text()[-6000:] if log.exists() else ''))
  if i==0:time.sleep(12)
 print('NATIVE_TERMINAL_SAMPLES',json.dumps(samples),flush=True)
print('HEALTH_COMPLETE',flush=True)
PY
