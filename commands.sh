#1 +60+a
#th2-78gg-BFG-resume-progress-20260929-a02
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-78gg-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/deep-kv-BFG-5000-20260929-a01')
print('PROGRESS_CHECK',datetime.now(timezone.utc).isoformat(),flush=True)
for name in ['production/supervisor.json','production/run/run.json']:
 print(name,(root/name).read_text(),flush=True)
path=root/'production/run/arm-B.log'
with path.open('rb') as f:
 f.seek(max(0,path.stat().st_size-16000))
 lines=f.read().decode(errors='replace').replace('\r','\n').splitlines()
print('B_LOG_TAIL',flush=True)
print('\n'.join([line for line in lines if line.strip()][-45:]),flush=True)
print('GPU_STATUS',json.dumps(snapshot(list(range(8)))),flush=True)
PY
