#1 +10+a
#th2-ohmv-b200-current-status-20261122-a01
set -eu
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD"
python3 - <<'PY'
import json
from scripts.gpu_process import inspect
s=inspect()
for record in s['processes'].values():
    record['command_prefix']=[x if x in ('/tmp/llm_pretrain_burn.py',) else '<other>' for x in record['command_prefix']]
print('B200_CURRENT_STATUS',json.dumps(s))
PY
