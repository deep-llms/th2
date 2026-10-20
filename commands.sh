#1 +30+a
#th2-6y3o-dot-pilot-preflight-20261020-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
export CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'PY'
from pathlib import Path
import json,datetime,shutil
from scripts.native_baseline_pipeline import burn_gate
from scripts.gpu_process import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2')
burn_gate(root/'6y3o-dot-dispatch-opt-20261020-a04')
assert json.loads((root/'6y3o-dot-dispatch-opt-20261020-a04/supervised/run/checks/summary.json').read_text())['status']=='passed'
print('PILOT_PREFLIGHT_UTC',datetime.datetime.now(datetime.timezone.utc).isoformat())
live=inspect();print('GPU_LIVE',json.dumps(live))
print('DISK_BYTES',shutil.disk_usage('/mnt/local')._asdict())
for f in ['assets-verified.json','cache-ready.json','prepared/baseline.json']:
 p=root/'6y3o-native12-cx-en-20261020-a02'/f;j=json.loads(p.read_text())
 if f=='assets-verified.json':j={k:j[k] for k in ['status','files_verified','data_root','subset_root','tokenizer','splits']}
 if f=='cache-ready.json':j={k:j[k] for k in ['status','seconds','train_sequences','validation_sequences']}
 print('BASELINE_ASSET',f,json.dumps(j))
assert not (root/'6y3o-dot-pilot-20261020-a01').exists()
print('PILOT_PREFLIGHT_PASSED')
PY
