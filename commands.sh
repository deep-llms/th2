#1 +120+a
#th2-ohmv-layer-adaptation-cpu-prepare-20261121-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
TASK_ROOT="$PWD"
TASK_BASE=/mnt/local/_outputs/deep-llms_th2/layer_adaptation_20261121_a01
"$TASK_PYTHON" - <<'PY'
import json,hashlib,shutil,socket
from pathlib import Path
from scripts.gpu_process import inspect,process
from scripts.train_then_burn import approved_launcher,GUARD_HASH
from efficient_llm.warm_order.data import sha
import torch,numpy
assert socket.gethostname()=='thiennh-p6-ohmv-worker-0'
assert torch.__version__=='2.14.1+cu130' and numpy.__version__=='2.4.6'
state=inspect();native='/tmp/llm_pretrain_burn.py';digest='3cdcc857bd01b096e20a02640fa85f0b8be7607e3c2b22a89a704bbac3650857'
assert len(state['workers'])==8 and not state['guard_disabled']
assert sha(native)==digest and sha('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
assert all(approved_launcher(process(p)['ppid'],{native:digest}) for p in state['workers'])
assert sha('resources/layer_adaptation_source_20261121.json')=='274d394e494521bf1333b1d63b428d0e2bf43f692fca66fb3e0e4abe12ae5824'
pin=json.loads(Path('resources/layer_adaptation_source_20261121.json').read_text())
for name,want in pin['files'].items():assert sha(name)==want,name
base=Path('/mnt/local/_outputs/deep-llms_th2/layer_adaptation_20261121_a01')
assert not base.exists(),'Existing output: reconcile, never overwrite'
base.mkdir(parents=True)
for name,want in pin['files'].items():
 dest=base/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest);assert sha(dest)==want
(base/'source-manifest.json').write_text(json.dumps(pin,indent=2)+'\n')
(base/'cpu-preownership.json').write_text(json.dumps(state,indent=2)+'\n')
print('ADAPTATION_CPU_START',json.dumps(dict(source_files=len(pin['files']),ownership=state)),flush=True)
PY
cd "$TASK_BASE/source"
export PYTHONPATH="$PWD:$PWD/tests/warm_order"
"$TASK_PYTHON" -m unittest discover -s tests/warm_order -p 'test_adaptation_prepare.py' -v
"$TASK_PYTHON" - <<'PY'
import json,torch
from pathlib import Path
from efficient_llm.warm_order.adaptation_prepare import prepare
from efficient_llm.warm_order.data import sha
torch.set_num_threads(4)
base=Path('/mnt/local/_outputs/deep-llms_th2/layer_adaptation_20261121_a01')
report=prepare('resources/layer_adaptation_reservation_node_20261121.json','ef8933f7a0aad3d8a3412ecddbec3965adbe1c3654a4c37c018912e9b170f4e9',
 'results/layer_dependency_20261120/adaptation_parent_registration.json',
 {str(s):f'/mnt/local/_outputs/deep-llms_th2/horizon_20261119/runs/D-s{s}/train' for s in (0,1)},
 [0,10,50,200,500,1000,2000],4,base/'prepared')
print('ADAPTATION_CPU_PREPARED',json.dumps(dict(report=report,prepared_sha256=sha(base/'prepared/manifest.json'),reservation_sha256=sha(base/'prepared/reservation/manifest.json'))),flush=True)
PY
cd "$TASK_ROOT"
export PYTHONPATH="$PWD"
"$TASK_PYTHON" - <<'PY'
import json
from pathlib import Path
from scripts.gpu_process import inspect,process,ownership
from scripts.train_then_burn import approved_launcher,GUARD_HASH
from efficient_llm.warm_order.data import sha
base=Path('/mnt/local/_outputs/deep-llms_th2/layer_adaptation_20261121_a01');state=inspect();prior=json.loads((base/'cpu-preownership.json').read_text())
assert ownership(state['gpus'])==ownership(prior['gpus']) and not state['guard_disabled']
assert all(process(int(pid))==p for pid,p in prior['processes'].items())
report=json.loads((base/'prepared/manifest.json').read_text())
assert report['actual_parents_loaded']==2 and report['train_jobs_launched']==0
proof=dict(complete=True,prepared_sha256=sha(base/'prepared/manifest.json'),ownership=state,train_jobs_launched=0,production_ready=False)
(base/'cpu-terminal.json').write_text(json.dumps(proof,indent=2)+'\n')
print('ADAPTATION_CPU_COMPLETE',json.dumps(proof),flush=True)
PY
