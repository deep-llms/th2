#1 +60+a
#th2-tjx3-full-attention-preflight-20261004-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/train_env/bin/python -u - <<'PY'
import hashlib,json
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
print('GPU_INSPECTION',json.dumps(inspect()))
recipe=json.loads(Path('deep_kv.b200.json').read_text())
manifest=json.loads(Path('resources/qwen3_base_assets.json').read_text())
root=Path(recipe['tokenizer_name'])
for name,expected in manifest['files'].items():
 p=root/name
 print('ASSET',str(p),p.is_file(), hashlib.sha256(p.read_bytes()).hexdigest()==expected if p.is_file() else None)
for parent in [Path('/mnt/local/_models/deep-llms_th2'),Path('/mnt/local/.cache/huggingface/hub')]:
 print('MODEL_ROOT',str(parent),sorted(p.name for p in parent.iterdir())[:30] if parent.is_dir() else 'missing')
p=Path(recipe['data_dir']);print('TRAIN_SPLIT',str(p),sorted(x.name for x in p.iterdir())[:6])
PY
