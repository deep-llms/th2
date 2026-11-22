#1 +120+a
#th2-ohmv-embedding-anchor-cpu-20261122-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH="$PWD:$PWD/tests/warm_order"
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
"$TASK_PYTHON" - <<'PY'
import json,shutil,socket
from pathlib import Path
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect
import torch,numpy
assert socket.gethostname()=='thiennh-p6-ohmv-worker-0'
assert torch.__version__=='2.14.1+cu130' and numpy.__version__=='2.4.6'
p=Path('resources/embedding_anchor_source_20261122.json');assert sha(p)=='07b12161ba3203f8beb2eec6c82e1f95530baf3f796cdbb1fb5e5cf858f82c11'
pins=json.loads(p.read_text())['files'];base=Path('/mnt/local/_outputs/deep-llms_th2/embedding_anchor_20261122_a01')
assert not base.exists(),'Existing output; reconcile before any retry'
for name,h in pins.items():assert sha(name)==h,name
base.mkdir(parents=True)
for name,h in pins.items():
 dest=base/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest);assert sha(dest)==h
(base/'source.json').write_text(json.dumps(pins,indent=2)+'\n')
(base/'cpu-preownership.json').write_text(json.dumps(inspect(),indent=2)+'\n')
print('ANCHOR_CPU_START',json.dumps(dict(source_pins=len(pins))),flush=True)
PY
cd /mnt/local/_outputs/deep-llms_th2/embedding_anchor_20261122_a01/source
export PYTHONPATH="$PWD:$PWD/tests/warm_order"
"$TASK_PYTHON" -m unittest discover -s tests/warm_order -p 'test_embedding_anchor*.py' -v
"$TASK_PYTHON" -m efficient_llm.warm_order.embedding_anchor_prepare --plan resources/embedding_anchor_pilot_20261122.json --sha d8fe493abf07508df5c3f922b92887516c14c26b07198e05d8385972463297d8 --out /mnt/local/_outputs/deep-llms_th2/embedding_anchor_20261122_a01/prepared
"$TASK_PYTHON" - <<'PYEND'
import json
from pathlib import Path
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect
base=Path('/mnt/local/_outputs/deep-llms_th2/embedding_anchor_20261122_a01')
r=dict(prepared_sha256=sha(base/'prepared/prepared.json'),prepared=json.loads((base/'prepared/prepared.json').read_text()),current_ownership=inspect())
assert r['prepared']['train_updates']==0 and not r['prepared']['gpu_admitted'] and not r['prepared']['fixture_only']
(base/'cpu-terminal.json').write_text(json.dumps(r,indent=2)+'\n')
print('ANCHOR_CPU_COMPLETE',json.dumps(r),flush=True)
PYEND
