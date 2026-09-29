#1 +60+a
#th2-78gg-perf-preflight-20260929-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-78gg-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/perf_env/bin/python - <<'PY'
import importlib.metadata as m, inspect, json, sys, torch
from flash_attn.cute import flash_attn_func
from datetime import datetime, timezone
from scripts.verified_gpu_reclaim import inspect as inspect_gpus
keys=['torch','transformers','accelerate','datasets','flash-attn-4','nvidia-cutlass-dsl','nvidia-cudnn-cu13','triton']
print('PERF_ENV',datetime.now(timezone.utc).isoformat(),sys.executable,json.dumps({k:m.version(k) for k in keys}),flush=True)
assert torch.__version__=='2.14.0+cu130',torch.__version__
assert m.version('transformers')=='5.9.0' and m.version('accelerate')=='1.13.0'
assert m.version('datasets')=='4.8.5' and m.version('flash-attn-4')=='4.0.0b32'
print('FA4_SIGNATURE',inspect.signature(flash_attn_func),flush=True)
print('GPU_INSPECTION',json.dumps(inspect_gpus()),flush=True)
print('CUDA_INITIALIZED',torch.cuda.is_initialized(),flush=True)
PY
nvidia-smi --query-gpu=index,name,driver_version,pstate,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu --format=csv
nvidia-smi topo -m
