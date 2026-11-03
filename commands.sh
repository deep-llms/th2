#1 +30+a
#th2-wc69-depth-plan-prepare-20261103-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
"$TASK_PYTHON" -u -m scripts.a2_depth_wc69_plan --reconstruction /mnt/local/_outputs/deep-llms_th2/wc69-a2-data-rebuild-20261103-a02/complete.json --restoration /mnt/local/_outputs/deep-llms_th2/wc69-a2-depth-assets-20261103-a02/complete.json --output /mnt/local/_outputs/deep-llms_th2/wc69-a2-depth-plan-20261103-a01/plan.json
"$TASK_PYTHON" -u - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
from importlib.metadata import version
import json,hashlib,base64
from scripts.a2_depth_wc69_admission import admission
p=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-depth-plan-20261103-a01/plan.json')
cfg=json.loads(p.read_text())
print('READONLY_ADMISSION',json.dumps(admission(cfg,Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-depth-partc-20261103-a01'))),flush=True)
print('TRAIN_RUNTIME',json.dumps({k:version(k) for k in ('torch','transformers','tokenizers','accelerate','datasets','numpy','pyarrow')}),flush=True)
print('PLAN_SHA256',hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
print('PLAN_BASE64',base64.b64encode(p.read_bytes()).decode(),flush=True)
print('PLAN_VERIFIED',datetime.now(timezone.utc).isoformat(),flush=True)
PY
