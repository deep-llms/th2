#1 +30+a
#th2-6y3o-dot-v3-status-20261021-a04
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'PY'
import json
from pathlib import Path
from scripts.gpu_process import inspect
p=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-v3-20261021-a01')
for name in ['supervised/run/run.json','supervised/run/complete.json','supervised/run/v3_acceptance.log','supervised/run/checks/summary.json','supervised/supervisor.json','supervised/burn-verified.json','supervisor.log']:
 f=p/name
 if f.is_file():print('V3_FILE',name,f.read_text(errors='replace')[-22000:],flush=True)
checks=p/'supervised/run/checks'
for f in sorted(checks.glob('*.log'),key=lambda f:f.stat().st_mtime)[-2:]:
 print('V3_CHECK_LOG',f.name,f.read_text(errors='replace')[-3000:],flush=True)
for f in sorted((p/'supervised/run').glob('*.attempt*.log'),key=lambda f:f.stat().st_mtime)[-1:]:
 print('TRAIN_LOG',f.name,f.read_text(errors='replace')[-9000:],flush=True)
print('GPU_LIVE',json.dumps(inspect()),flush=True)
PY
