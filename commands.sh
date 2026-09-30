#1 +60+a
#th2-78gg-bottleneck-preflight-20260930-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-78gg-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python -u - <<'PY'
import hashlib, importlib.metadata, json, shutil
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
from scripts.train_then_burn import approved_launcher, APPROVED_BURNS, BURN_HASH, GUARD_HASH, process
inspection = inspect()
print('GPU_INSPECTION', json.dumps(inspection), flush=True)
approved = {**APPROVED_BURNS, str(Path('resources/llm_pretrain_burn.py').resolve()): BURN_HASH}
assert not inspection['guard_disabled']
assert all(approved_launcher(pid, approved) or approved_launcher(process(pid)['ppid'], approved)
           for pid in inspection['workers'])
assert hashlib.sha256(Path('/mnt/local/_gpu_guard/gpu_guard.sh').read_bytes()).hexdigest() == GUARD_HASH
for name in ('torch', 'transformers', 'accelerate', 'datasets'):
 print('ENV_VERSION', name, importlib.metadata.version(name), flush=True)
recipe = json.loads(Path('deep_kv.b200.json').read_text())
for key in ('config_name', 'tokenizer_name', 'data_dir', 'eval_data_dir'):
 path = Path(recipe[key]); assert path.exists(), str(path)
 print('INPUT_EXISTS', key, str(path), flush=True)
print('FREE_DISK_GIB', shutil.disk_usage('/mnt/local/_outputs').free / 2**30, flush=True)
print('READ_ONLY_PREFLIGHT_PASSED', flush=True)
PY
