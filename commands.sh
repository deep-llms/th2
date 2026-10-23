#1 +30+a
#th2-6y3o-dot-stop-preflight-20261023-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import json, importlib.metadata as m, hashlib, subprocess
from pathlib import Path
from scripts.gpu_process import inspect
from scripts.native_baseline_pipeline import validate_burn_handoff
from scripts.train_then_burn import burn_progress
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-downstream-20261023-a01')
read=lambda p:json.loads(p.read_text())
live=inspect();state=read(root/'supervised/supervisor.json');proof=read(root/'supervised/burn-verified.json')
validate_burn_handoff(state,proof,live)
assert burn_progress((root/'supervised/burn.log').read_text())
parent=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-long-pair-20261023-a01')
recipes={a:read(parent/'prepared'/(a+'-s0.train.json')) for a in ('D','Partial-25-cosine')}
print('STOP_PREFLIGHT '+json.dumps(dict(live=live,recipes=recipes,previous_report=read(parent/'supervised/run/long-pair-report.json'))),flush=True)
REMOTE
