#1 +30+a
#th2-6y3o-joint-groups-preflight-20261026-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import hashlib, importlib.metadata as meta, json, socket, subprocess
from pathlib import Path
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
print('PREFLIGHT_START joint-groups-a01',flush=True)
print('ENV '+json.dumps({n:meta.version(n) for n in ['torch','transformers','datasets','accelerate','pyarrow']}),flush=True)
model=Path('/mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6')
print('MODEL '+json.dumps({'path':str(model),'config':json.loads((model/'config.json').read_text()),'files':[p.name for p in model.iterdir()]}),flush=True)
raw=Path('/mnt/local/_data/deep-llms_th2/raw')
for p in raw.iterdir():
    print('DATA '+json.dumps({'path':str(p),'children':[q.name for q in p.iterdir()] if p.is_dir() else []}),flush=True)
for p in [Path('/mnt/local/_gpu_guard/gpu_guard.sh'),Path('/mnt/local/_gpu_guard/DISABLED')]:
    print('GUARD '+json.dumps({'path':str(p),'exists':p.exists(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None}),flush=True)
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used','--format=csv,noheader'],text=True),flush=True)
print(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True),flush=True)
for name in ['6y3o-dot-pilot-20261020-a01','6y3o-native12-cx-en-20261020-a02']:
    p=Path('/mnt/local/_outputs/deep-llms_th2')/name/'assets-verified.json'
    if p.is_file():
        obj=json.loads(p.read_text()); print('PREVIOUS_ASSETS '+json.dumps({k:obj.get(k) for k in ['status','data_root','subset_root','tokenizer']}),flush=True)
print('PREFLIGHT_COMPLETE',flush=True)
REMOTE
