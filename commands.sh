#1 +30+a
#th2-6y3o-efficient-kv-downstream-preflight-20261028-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 -u - <<'READONLY'
import json, socket, subprocess, hashlib
from pathlib import Path
from datetime import datetime, timezone
from scripts.gpu_process import inspect
from eval.benchmark_data import local_dataset_paths
assert socket.gethostname() == 'thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-long-20261027-a01')
print('PREFLIGHT_AT', datetime.now(timezone.utc).isoformat(), flush=True)
print('GPU_INSPECTION', json.dumps(inspect()), flush=True)
for rel in ['supervised/supervisor.json','supervised/burn-verified.json','supervised/run/complete.json','supervised/run/summary.json']:
 print('PARENT_ARTIFACT',json.dumps(dict(path=rel,value=json.loads((root/rel).read_text()))),flush=True)
for arm in ['A','B']:
 cp=root/'supervised/run'/arm/'final'
 r=json.loads((cp.parent/'result.json').read_text())
 files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in cp.iterdir() if p.is_file()}
 print('CHECKPOINT',json.dumps(dict(arm=arm,files=files,result=r)),flush=True)
data=Path('/mnt/local/_data/deep-llms_th2/benchmarks_zero_20261023')
print('DATA_VERIFIED',json.dumps(local_dataset_paths(data,'resources/downstream_english_20261023.json')),flush=True)
for name in ['train','eval']:
 py=Path('/mnt/local/conda-py311/envs')/name/'bin/python3.11'
 if py.is_file():
  result=subprocess.run([str(py),'-c',"import json,importlib.metadata as m; from lm_eval.models.huggingface import HFLM; print(json.dumps({k:m.version(k) for k in ['torch','transformers','datasets','accelerate','lm-eval','tokenizers','safetensors']}))"],text=True,capture_output=True)
  print('ENVIRONMENT',json.dumps(dict(name=name,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr[-4000:])),flush=True)
print('PREFLIGHT_COMPLETE',flush=True)
READONLY
