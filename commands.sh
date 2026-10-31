#1 +30+a
#th2-6y3o-a2-mech-health-20261031-a03
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import hashlib,json,socket,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect,process
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-mech-20261031-a01')
base=root/'results/a2_mech_20261031'
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
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
  if len(raw)<10000000:print('ARTIFACT',json.dumps(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
  else:print('LARGE_ARTIFACT',json.dumps(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))),flush=True)
 else:print('MISSING',str(p.relative_to(root)),flush=True)
for rel in ('source-manifest.json','jobs.json','budget.json','admission.json','accelerate-env.log',
 'supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/burn-verified.json',
 'supervised/run/run.json','supervised/run/complete.json','supervised/run/mech-complete.json'):
 export(root/rel)
for rel in ('config/plan.json','preparation_checks.json','progress.json','complete.json','coordinator.json','REPORT.md'):
 export(base/rel)
manifest=root/'source-manifest.json'
if manifest.exists():
 m=json.loads(manifest.read_text());changed=[name for name,value in m['files'].items() if hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()!=value]
 print('SOURCE_VERIFICATION',json.dumps(dict(files=len(m['files']),code_commit=m['code_commit'],changed=changed)),flush=True)
for path in sorted(base.glob('workers/*/*/*/tests.json')):
 rows=json.loads(path.read_text());bad=[r for r in rows if not r['passed']]
 print('WORKER_TEST_SUMMARY',json.dumps(dict(path=str(path.relative_to(root)),checks=len(rows),flagged=len(bad),recent_flags=bad[-3:])),flush=True)
for path in sorted(base.glob('workers/*/*/*/*.json')):
 if path.name in ('tests.json','result.json'):continue
 value=json.loads(path.read_text())
 print('MEASUREMENT',json.dumps(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),loss=value.get('loss'),sequences=value.get('sequences'),unverified=value.get('unverified'))),flush=True)
for path in sorted(base.glob('workers/*/*/*/result.json')):
 value=json.loads(path.read_text())
 print('WORKER_COMPLETE',json.dumps(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),elapsed_seconds=value['elapsed_seconds'],execution_complete=value['execution_complete'])),flush=True)
for rel in ('cpu_completion/admission.json','cpu_completion/progress.json','cpu_completion/complete.json','cpu_completion/source-manifest.json'):
 export(root/rel)
for rel in ('readout_frequency_revision/verification.json','readout_frequency_revision/decision_mech.json','readout_frequency_revision/REPORT.md'):
 export(base/rel)
cpum=root/'cpu_completion/source-manifest.json'
if cpum.exists():
 m=json.loads(cpum.read_text());changed=[name for name,value in m['files'].items() if hashlib.sha256((root/'cpu_completion/source'/name).read_bytes()).hexdigest()!=value]
 print('CPU_SOURCE_VERIFICATION',json.dumps(dict(files=len(m['files']),code_commit=m['code_commit'],changed=changed)),flush=True)
logs=[root/'cpu_completion_launch.log',*root.glob('cpu_completion/*.log'),root/'supervisor.log',root/'supervised/burn.log',root/'supervised/run/a2-mech.log',*base.glob('logs/**/*.log')]
for p in logs:
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2000));tail=f.read().decode(errors='replace')
  print('LOG_TAIL',json.dumps(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,text=tail)),flush=True)
guard=Path('/mnt/local/_gpu_guard/DISABLED');print('GUARD',guard.read_text() if guard.is_file() else 'released',flush=True)
print('SESSION',subprocess.run(['tmux','display-message','-p','-t','a2_mech_20261031_a01','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip(),flush=True)
rows=subprocess.run(['ps','-eo','pid,ppid,pcpu,pmem,etime,args'],capture_output=True,text=True,check=True).stdout.splitlines()
print('LAUNCH_PROCESSES',json.dumps([r for r in rows if 'a2_mech' in r or str(root) in r]),flush=True)
terminal=root/'supervised/supervisor.json'
if terminal.exists() and json.loads(terminal.read_text()).get('finished_at'):
 samples=[]
 for i in range(2):
  snap=inspect();proc={}
  for pid in snap['workers']:
   item=process(pid);proc[str(pid)]=item
   if item['ppid']>1:proc[str(item['ppid'])]=process(item['ppid'])
  log=root/'supervised/burn.log';tail=log.read_text()[-6000:] if log.exists() else ''
  samples.append(dict(inspection=snap,processes=proc,burn_tail=tail))
  if i==0:time.sleep(12)
 print('NATIVE_TERMINAL_SAMPLES',json.dumps(samples),flush=True)
print('HEALTH_COMPLETE',flush=True)
PY
