#1 +60+a
#th2-q359-smoke-preflight-20261009-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import json
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
from scripts.train_then_burn import APPROVED_BURNS, GUARD_HASH, digest, approved_launcher
print('INSPECTION', json.dumps(inspect()), flush=True)
for path, expected in {**APPROVED_BURNS, '/mnt/local/_gpu_guard/gpu_guard.sh': GUARD_HASH}.items():
    p=Path(path)
    print('SOURCE', path, 'exists', p.is_file(), 'sha256', digest(p) if p.is_file() else None, 'expected', expected, flush=True)
record=inspect()
print('BURN_IDENTIFIED', {str(pid): (approved_launcher(pid,APPROVED_BURNS) or (record['processes'][str(pid)]['ppid']>1 and approved_launcher(record['processes'][str(pid)]['ppid'],APPROVED_BURNS))) for pid in record['workers']}, flush=True)
recipe=json.loads(Path('proxy_heads.b200.json').read_text())
print('ASSETS', {key:Path(recipe[key]).exists() for key in ('config_name','tokenizer_name','data_dir','eval_data_dir')}, flush=True)
print('SMOKE_PREFLIGHT_COMPLETE', flush=True)
PYREMOTE
