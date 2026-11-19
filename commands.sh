#1 +120+a
#th2-ohmv-soft-exploration-cpu-proof-20261119-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
import json,time
from pathlib import Path
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect,process,ownership
from scripts.train_then_burn import approved_launcher,GUARD_HASH
base=Path('/mnt/local/_outputs/deep-llms_th2/soft_exploration_20261119_a01')
pin=json.loads(Path('resources/soft_exploration_source_20261119.json').read_text())['files']
r=dict(utc=time.time(),source_files=len(pin),source_actual={p:sha(p)==h for p,h in pin.items()},ownership=inspect(),ready=False)
assert all(r['source_actual'].values())
if (base/'inputs/complete.json').exists() and (base/'cpu-ownership.json').exists():
 prepared=json.loads((base/'inputs/complete.json').read_text());prior=json.loads((base/'cpu-ownership.json').read_text())
 r.update(prepared=prepared,prepared_sha256=sha(base/'inputs/complete.json'))
 r['input_actual']={p:sha(p)==h for p,h in prepared['hashes'].items()}
 r['prepared_actual']={p:sha(base/'inputs'/p)==h for p,h in prepared['files'].items()}
 assert all(r['input_actual'].values()) and all(r['prepared_actual'].values())
 assert ownership(prior['gpus'])==ownership(r['ownership']['gpus']) and all(process(int(pid))==rec for pid,rec in prior['processes'].items())
 native='/tmp/llm_pretrain_burn.py';digest='3cdcc857bd01b096e20a02640fa85f0b8be7607e3c2b22a89a704bbac3650857'
 assert sha(native)==digest and sha('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
 assert not r['ownership']['guard_disabled'] and len(r['ownership']['workers'])==8
 assert all(approved_launcher(process(pid)['ppid'],{native:digest}) for pid in r['ownership']['workers'])
 r['configs']={name:json.loads((base/'inputs'/name).read_text()) for name in prepared['files']}
 r['ready']=True
print('SOFT_EXPLORATION_CPU_PROOF',json.dumps(r),flush=True)
PY
