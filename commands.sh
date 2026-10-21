#1 +30+a
#th2-6y3o-dot-v3-preflight-20261021-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 /mnt/local/conda-py311/envs/train/bin/python3.11 - <<'PY'
import json,shutil
from pathlib import Path
from scripts.gpu_process import inspect
from scripts.native_baseline_pipeline import burn_gate
from scripts.train_then_burn import burn_progress,digest,GUARD_HASH,BURN_HASH
base=Path('/mnt/local/_outputs/deep-llms_th2')
pred=base/'6y3o-dot-diagnostics-gradient-20261021-a02'
root=base/'6y3o-dot-v3-20261021-a01'
burn_gate(pred)
assert burn_progress((pred/'supervised/burn.log').read_text())
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
assert digest('resources/llm_pretrain_burn.py')==BURN_HASH
assert not root.exists()
pilot=base/'6y3o-dot-pilot-20261020-a01'
recipe=json.loads((pilot/'prepared/runs/P1-A/training_recipe.json').read_text())
for name in ['bdot/model.py','bdot/routing.py','bdot/grouped.py','bdot/trainer.py']:
 assert digest(name)==recipe['code_sha256'][name],name
free=shutil.disk_usage(base).free
assert free>300*1024**3,free
root.mkdir()
shutil.copytree('bdot',root/'reference/bdot',ignore=shutil.ignore_patterns('__pycache__'))
receipt={'status':'ok','free_gib':free/1024**3,'live':inspect(),
 'reference_hashes':{str(p.relative_to(root/'reference')):digest(p) for p in (root/'reference/bdot').glob('*.py')}}
(root/'initial-preflight.json').write_text(json.dumps(receipt,indent=2))
print('V3_PREFLIGHT',json.dumps(receipt),flush=True)
print('V3_PREFLIGHT_OK',flush=True)
PY
