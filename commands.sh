#1 +60+a
#th2-ohmv-local-order-cpu-admission-20261118-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
TASK_BASE=/mnt/local/_outputs/deep-llms_th2/local_order_20261118
mkdir -p "$TASK_BASE"
"$TASK_PYTHON" -m unittest discover -s tests/warm_order -p 'test_local*.py' -v
"$TASK_PYTHON" -m efficient_llm.warm_order.local_prepare --plan resources/local_order_data_20261118.json --out "$TASK_BASE/inputs"
"$TASK_PYTHON" - <<'PYCODE'
import json,sys,socket
from pathlib import Path
import torch,numpy
from scripts.gpu_process import inspect
from efficient_llm.warm_order.train import write_json
from efficient_llm.warm_order.data import sha
base=Path('/mnt/local/_outputs/deep-llms_th2/local_order_20261118')
assert socket.gethostname()=='thiennh-p6-ohmv-worker-0'
x=inspect();write_json(base/'cpu-ownership.json',x)
print('CPU_ADMISSION',json.dumps(dict(prepared_sha256=sha(base/'inputs/complete.json'),ownership=x,torch=torch.__version__,numpy=numpy.__version__,executable=sys.executable)))
PYCODE
