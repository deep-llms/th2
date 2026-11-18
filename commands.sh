#1 +60+a
#th2-ohmv-warm-order-cpu-admission-20261118-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
TASK_PY=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
TASK_BASE=/mnt/local/_outputs/deep-llms_th2/warm_order_20261118
[ "$(hostname)" = thiennh-p6-ohmv-worker-0 ]
[ ! -e "$TASK_BASE" ]
"$TASK_PY" - <<'PYCODE'
from pathlib import Path
import hashlib,json,shutil
root=Path.cwd();base=Path('/mnt/local/_outputs/deep-llms_th2/warm_order_20261118');base.mkdir()
files=[*root.glob('efficient_llm/warm_order/*.py'),*root.glob('tests/warm_order/*.py')]
files += [root/p for p in ['efficient_llm/__init__.py','efficient_llm/dot_pool/__init__.py','efficient_llm/dot_pool/model.py','scripts/gpu_process.py','scripts/gpu_status.py','resources/warm_order_data_20261118.json','docs/WARM_ORDER_PROTOCOL_20261118.md']]
hashes={}
for p in files:
 rel=p.relative_to(root);out=base/'cpu-source'/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out);hashes[str(rel)]=hashlib.sha256(out.read_bytes()).hexdigest()
(base/'cpu-source.json').write_text(json.dumps(hashes,indent=2))
PYCODE
cd "$TASK_BASE/cpu-source"
export PYTHONPATH="$PWD"
"$TASK_PY" -m pip check
"$TASK_PY" -m unittest discover -s tests/warm_order -v > "$TASK_BASE/cpu-tests.log" 2>&1
tail -n 5 "$TASK_BASE/cpu-tests.log"
"$TASK_PY" -m efficient_llm.warm_order.prepare --plan resources/warm_order_data_20261118.json --out "$TASK_BASE/inputs"
"$TASK_PY" - <<'PYCODE'
import json,hashlib
from pathlib import Path
from scripts.gpu_process import inspect
base=Path('/mnt/local/_outputs/deep-llms_th2/warm_order_20261118');x=inspect()
x['guard_dir_exists']=Path('/mnt/local/_gpu_guard').exists()
x['native_burn_sha256']=hashlib.sha256(Path('/tmp/llm_pretrain_burn.py').read_bytes()).hexdigest()
(base/'cpu-ownership.json').write_text(json.dumps(x,indent=2));print('OWNERSHIP',json.dumps(x))
print('WARM_ORDER_CPU_READY',hashlib.sha256((base/'inputs/complete.json').read_bytes()).hexdigest())
PYCODE
