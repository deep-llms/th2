#1 +30+a
#th2-q359-template-smoke-preflight-20261017-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python -u - <<'PY'
import json, subprocess
from pathlib import Path
from datetime import datetime, timezone
from scripts.verified_gpu_reclaim import inspect
print('TIME',datetime.now(timezone.utc).isoformat())
print('INSPECTION',json.dumps(inspect()))
for env in ('train_env','train_fa4','eval_fa4'):
    p=Path('/mnt/local/conda-py311/envs')/env/'bin/python'
    if p.exists():
        r=subprocess.run([str(p),'-c',"import sys,importlib.metadata as m; print(sys.executable); print({n:m.version(n) for n in ('torch','transformers','accelerate','datasets','flash-attn-4')})"],capture_output=True,text=True)
        print('ENV',env,r.returncode,r.stdout,r.stderr)
for path in ('/mnt/local/_gpu_guard/DISABLED','/mnt/local/_outputs/deep-llms_th2/q359-ce7454-soups-20261017-a01/supervised/supervisor.json'):
    p=Path(path); print('STATE',path,p.read_text() if p.exists() else 'ABSENT')
print('TEMPLATE_PREFLIGHT_COMPLETE')
PY
