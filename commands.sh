#1 +120+a
#th2-ohmv-multi-exit-cpu-admission-20261119-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
"$TASK_PYTHON" - <<'PY'
import json,socket,torch,numpy as np
from pathlib import Path
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect,process
from scripts.train_then_burn import approved_launcher,GUARD_HASH
assert socket.gethostname()=='thiennh-p6-ohmv-worker-0'
assert torch.__version__=='2.14.1+cu130' and np.__version__=='2.4.6'
pin=json.loads(Path('resources/multi_exit_source_20261119.json').read_text())
for p,want in pin['files'].items():assert sha(p)==want,p
state=inspect();assert len(state['workers'])==8 and not state['guard_disabled']
native='/tmp/llm_pretrain_burn.py';digest='3cdcc857bd01b096e20a02640fa85f0b8be7607e3c2b22a89a704bbac3650857'
assert sha(native)==digest
assert sha('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
assert all(approved_launcher(process(p)['ppid'],{native:digest}) for p in state['workers'])
base=Path('/mnt/local/_outputs/deep-llms_th2/multi_exit_20261119')
assert not base.exists(),'Existing output; reconcile, never overwrite'
print('CPU_ADMISSION_PRECHECK',json.dumps(dict(source_files=len(pin['files']),ownership=state)),flush=True)
PY
"$TASK_PYTHON" -m unittest tests.warm_order.test_multi_exit tests.warm_order.test_multi_exit_train tests.warm_order.test_multi_exit_pipeline tests.warm_order.test_multi_exit_reservation -v
"$TASK_PYTHON" -m efficient_llm.warm_order.multi_exit_prepare --out /mnt/local/_outputs/deep-llms_th2/multi_exit_20261119/inputs --plan resources/multi_exit_plan_20261119.json
"$TASK_PYTHON" - <<'PY'
import json
from pathlib import Path
from scripts.gpu_process import inspect
from efficient_llm.warm_order.train import write_json
base=Path('/mnt/local/_outputs/deep-llms_th2/multi_exit_20261119');state=inspect()
write_json(base/'cpu-ownership.json',state)
print('MULTI_EXIT_CPU_TERMINAL',json.dumps(state),flush=True)
PY
