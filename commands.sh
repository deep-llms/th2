#1 +30+a
#th2-q359-dot-resume-preflight-20261018-a01
set -euo pipefail
TASK_PYTHON=/mnt/local/conda-py311/envs/eval_fa4/bin/python
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" scripts/bdot_b200_preflight.py --output /mnt/local/_outputs/deep-llms_th2/q359-dot-resume-20261018-a01/preflight.json
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" - <<'PY'
import inspect
from flash_attn.cute.interface import flash_attn_varlen_func
print('FA4_SIGNATURE',inspect.signature(flash_attn_varlen_func))
source=inspect.getfile(flash_attn_varlen_func)
for i,line in enumerate(open(source),1):
    if 'deterministic' in line:
        print('FA4_DETERMINISM',i,line.rstrip())
PY
