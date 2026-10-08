#1 +60+a
#th2-tjx3-proxy-validation-inspect-20261008-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
date -u
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - <<'CHECK'
import json,sys,shutil,importlib.metadata as md
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
print('INSPECTION',json.dumps(inspect()),flush=True)
print('ENVIRONMENT',json.dumps({p:md.version(p) for p in ('torch','transformers','accelerate','datasets','flash-attn-4','nvidia-cutlass-dsl')}),flush=True)
print('FREE_GIB',shutil.disk_usage('/mnt/local').free/2**30,flush=True)
for p in ('/mnt/local/_outputs/deep-llms_th2/supervised-stsb-boolq-20261008-a03/supervised/supervisor.json','/mnt/local/_outputs/deep-llms_th2/proxy-final-attention-check-20261005-a01/prepared/train.json'):
 f=Path(p);print('INPUT',p,f.is_file(),f.stat().st_size if f.is_file() else None,flush=True)
 if f.name=='supervisor.json' and f.is_file():print('PREVIOUS_HANDOFF',f.read_text(),flush=True)
print('INSPECTION_FINISHED',flush=True)
CHECK
