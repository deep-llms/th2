#1 +30+a
#th2-q359-dot-grouped-status-20261018-a01
set -euo pipefail
TASK_PYTHON=/mnt/local/conda-py311/envs/eval_fa4/bin/python
"$TASK_PYTHON" - <<'CHECK'
import json
from pathlib import Path
from scripts.gpu_process import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-dot-grouped-20261018-a03')
for name in ['supervised/supervisor.json','supervised/run/checks/summary.json']:
 p=root/name
 print('GROUPED_STATUS '+json.dumps(dict(path=name,value=json.loads(p.read_text()) if p.exists() else None)),flush=True)
for name in ['supervisor.log','supervised/run/grouped_verification.log','supervised/run/checks/resume_A.log','supervised/run/checks/resume_B.log']:
 p=root/name
 print('GROUPED_LOG '+json.dumps(dict(path=name,tail=p.read_text()[-5000:] if p.exists() else None)),flush=True)
s=inspect()
print('GPU_STATUS '+json.dumps(dict(host=s['host'],guard_disabled=s['guard_disabled'],gpus=s['gpus'])),flush=True)
CHECK
