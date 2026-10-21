#1 +30+a
#th2-6y3o-dot-v3-status-20261021-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'PY'
import json
from pathlib import Path
from scripts.gpu_process import inspect
p=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-v3-20261021-a01')
for name in ['prepare.log','preparation-failed.json','launch.json','accelerate_env.txt','supervisor.log',
 'supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/run.json',
 'supervised/run/v3_acceptance.log','supervised/run/checks/summary.json',
 'supervised/supervisor.json','supervised/burn-verified.json']:
 f=p/name
 if f.is_file():print('V3_FILE',name,f.read_text(errors='replace')[-14000:],flush=True)
checks=p/'supervised/run/checks'
if checks.exists():
 logs=sorted(checks.glob('*.log'),key=lambda f:f.stat().st_mtime)[-2:]
 for f in logs:print('V3_CHECK_LOG',f.name,f.read_text(errors='replace')[-5000:],flush=True)
print('GPU_LIVE',json.dumps(inspect()),flush=True)
PY
