#0
#th2-q359-expanded-paired-status-20261012-a06
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
python3 -u - <<'PY'
import datetime,hashlib,json,subprocess
from pathlib import Path
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-expanded-paired-20k-20261012-a03');run=root/'supervised/run'
print('OBSERVED_UTC',datetime.datetime.now(datetime.timezone.utc).isoformat())
for rel in ('supervised/supervisor.json','supervised/run/run.json','supervised/run/finetune/run.json'):
 p=root/rel
 if p.exists():
  value=json.loads(p.read_text())
  if 'jobs' in value:
   jobs=value['jobs'];value={**value,'jobs':jobs if len(jobs)<15 else jobs[-8:],'recorded_jobs':len(jobs),'successful_jobs':sum(j.get('status')=='ok' for j in jobs)}
  print('STATE',rel,json.dumps(value),flush=True)
for rel in ('supervised/run/complete.json','supervised/run/expanded-smoke/summary.json','supervised/run/expanded-full/summary.json','supervised/run/finetune/summary.json','supervised/run/paired-F.json','supervised/run/paired-Z.json','supervised/run/paired-probes.json','supervised/run/paired-position.json'):
 p=root/rel
 if p.exists():
  b=p.read_bytes();v=json.loads(b);print('ARTIFACT',rel,'bytes',len(b),'sha256',hashlib.sha256(b).hexdigest(),'status',v.get('status'),flush=True)
for rel in ('supervisor.log','supervised/burn.log'):
 p=root/rel
 if p.exists():print('LOG',rel,'\n'.join(p.read_text().splitlines()[-25:]),flush=True)
for folder in (run,run/'finetune'):
 logs=sorted(folder.glob('*.log'),key=lambda p:p.stat().st_mtime)
 for p in logs[-3:]:print('JOB_LOG',str(p.relative_to(root)),'\n'.join(p.read_text().splitlines()[-14:]),flush=True)
for p in sorted((run/'expanded-full').glob('*/*.log')):
 print('BENCH_LOG',str(p.relative_to(root)),'\n'.join(p.read_text().splitlines()[-5:]),flush=True)
subprocess.run(['tmux','list-panes','-t',root.name,'-F','#{pane_pid} #{pane_dead} #{pane_current_command}'],check=True)
subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv'],check=True)
subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],check=True)
PY
