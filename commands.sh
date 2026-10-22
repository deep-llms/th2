#1 +30+a
#th2-6y3o-dot-mlp-preflight-20261022-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 /mnt/local/conda-py311/envs/train/bin/python3.11 - <<'PY'
import json,shutil
from pathlib import Path
from scripts.gpu_process import inspect
from scripts.native_baseline_pipeline import burn_gate
from scripts.train_then_burn import burn_progress,digest,GUARD_HASH,BURN_HASH
base=Path('/mnt/local/_outputs/deep-llms_th2')
pred=base/'6y3o-dot-C1-20261022-a01'
root=base/'6y3o-dot-mlp-20261022-a01'
read=lambda p:json.loads(p.read_text())
assert (pred/'supervised/run/complete.json').is_file()
assert read(pred/'supervised/run/C1-summary.json')['status']=='ok'
burn_gate(pred)
assert burn_progress((pred/'supervised/burn.log').read_text())
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
assert digest('resources/llm_pretrain_burn.py')==BURN_HASH
assert not root.exists()
recipe=read(pred/'prepared/runs/C1-s0/training_recipe.json')
for name,expected in recipe['code_sha256'].items():
 assert digest(name)==expected, name
free=shutil.disk_usage(base).free
assert free>300*1024**3,free
root.mkdir()
shutil.copytree('bdot',root/'reference/bdot',ignore=shutil.ignore_patterns('__pycache__'))
receipt={'status':'ok','free_gib':free/1024**3,'live':inspect(),
 'training_code_sha256':recipe['code_sha256'],'predecessor':str(pred)}
(root/'initial-preflight.json').write_text(json.dumps(receipt,indent=2))
print('MLP_PREFLIGHT',json.dumps(receipt),flush=True)
print('MLP_PREFLIGHT_OK',flush=True)
PY
