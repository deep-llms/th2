#1 +30+a
#th2-q359-dot-grouped-status-20261018-a02

set -euo pipefail
printf '%s\n' 'DOT_RUNNER_PING_20261018_2254'
date -u +'%Y-%m-%dT%H:%M:%SZ'
hostname
pwd
sha256sum commands.sh
TASK_PYTHON=/mnt/local/conda-py311/envs/eval_fa4/bin/python
"$TASK_PYTHON" - <<'CHECK'
import json,subprocess
from pathlib import Path
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-dot-grouped-20261018-a03')
for name in ['supervised/supervisor.json','supervised/run/checks/summary.json']:
 p=root/name
 try:
  value=json.loads(p.read_text()) if p.exists() else None
 except (OSError,ValueError) as e:
  value={'read_error':str(e)}
 print('GROUPED_STATUS '+json.dumps(dict(path=name,value=value)),flush=True)
for name in ['supervisor.log','supervised/run/grouped_verification.log','supervised/run/checks/resume_A.log','supervised/run/checks/resume_B.log']:
 p=root/name
 if p.is_file():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-5000))
   tail=f.read(5000).decode(errors='replace')
 else:tail=None
 print('GROUPED_LOG '+json.dumps(dict(path=name,tail=tail)),flush=True)
for args in [['--query-gpu=index,name,utilization.gpu,memory.used','--format=csv'],['--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv']]:
 try:
  r=subprocess.run(['nvidia-smi',*args],capture_output=True,text=True,timeout=20)
  print('GPU_QUERY '+json.dumps(dict(args=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)),flush=True)
 except subprocess.TimeoutExpired:
  print('GPU_QUERY_TIMEOUT '+json.dumps(args),flush=True)
print('DOT_RUNNER_PING_COMPLETE_20261018_2254',flush=True)
CHECK
