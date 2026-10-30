#1 +30+a
#th2-6y3o-a2-quant-health-20261031-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import hashlib,json,socket,subprocess
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01')
base=root/'results/a2_quant_20261031'
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
def export(p):
 if p.is_file():
  raw=p.read_bytes()
  if len(raw)<10000000:print('ARTIFACT',json.dumps(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
  else:print('LARGE_ARTIFACT',json.dumps(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))),flush=True)
 else:print('MISSING',str(p.relative_to(root)),flush=True)
for rel in ('source-manifest.json','jobs.json','budget.json','admission.json','accelerate-env.log',
 'supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/burn-verified.json',
 'supervised/run/run.json','supervised/run/complete.json','supervised/run/quant-complete.json'):
 export(root/rel)
for rel in ('config/plan.json','preparation_checks.json','manifests/gptq_implementation.json','pooled_downstream.json',
 'labels_note.json','decision_quant.json','bitwidth.json','downstream_paired.json','downstream.json','tests.json','complete.json','REPORT.md','runs.jsonl'):
 export(base/rel)
manifest=root/'source-manifest.json'
if manifest.exists():
 m=json.loads(manifest.read_text());changed=[name for name,value in m['files'].items() if hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()!=value]
 print('SOURCE_VERIFICATION',json.dumps(dict(files=len(m['files']),code_commit=m['code_commit'],changed=changed)),flush=True)
for path in sorted(base.glob('workers/*/*/*/complete.json')):export(path)
for path in sorted(base.glob('workers/*/*/*/tests.json')):
 rows=json.loads(path.read_text());bad=[r for r in rows if not r['passed']]
 print('WORKER_TEST_SUMMARY',json.dumps(dict(path=str(path.relative_to(root)),checks=len(rows),flagged=len(bad),recent_flags=bad[-3:])),flush=True)
for path in sorted(base.glob('workers/*/*/*/runs.jsonl')):
 print('WORKER_RUNS',json.dumps(dict(path=str(path.relative_to(root)),rows=[json.loads(line) for line in path.read_text().splitlines()])),flush=True)
logs=[root/'supervisor.log',root/'supervised/burn.log',root/'supervised/run/a2-quant.log',*base.glob('logs/**/*.log')]
for p in logs:
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2000));tail=f.read().decode(errors='replace')
  print('LOG_TAIL',json.dumps(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,text=tail)),flush=True)
guard=Path('/mnt/local/_gpu_guard/DISABLED');print('GUARD',guard.read_text() if guard.is_file() else 'released',flush=True)
print('SESSION',subprocess.run(['tmux','display-message','-p','-t','a2_quant_20261031_a01','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip(),flush=True)
rows=subprocess.run(['ps','-eo','pid,ppid,pcpu,pmem,etime,args'],capture_output=True,text=True,check=True).stdout.splitlines()
print('LAUNCH_PROCESSES',json.dumps([r for r in rows if 'a2_quant' in r or str(root) in r]),flush=True)
print('HEALTH_COMPLETE',flush=True)
PY
