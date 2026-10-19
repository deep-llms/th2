#1 +30+a
#th2-6y3o-attention-env-inspect-20261019-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1
python3 - <<'CHECK'
import datetime, json, os, socket, subprocess
from pathlib import Path
print('RUNTIME_HOST '+json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),cwd=os.getcwd())),flush=True)
probe = r"""
import importlib, json, sys
from importlib.metadata import version, PackageNotFoundError
result=dict(python=sys.version, executable=sys.executable, prefix=sys.prefix, versions={}, imports={})
for name in ('torch','transformers','accelerate','datasets','tokenizers','safetensors','flash-attn','flash-attn-4','nvidia-cutlass-dsl','quack-kernels','einops','lm-eval','nvidia-cuda-runtime-cu13','nvidia-cudnn-cu13'):
    try: result['versions'][name]=version(name)
    except PackageNotFoundError: result['versions'][name]=None
for name in ('torch','transformers','accelerate','datasets','tokenizers','safetensors'):
    try:
        module=importlib.import_module(name)
        result['imports'][name]='ok'
        if name=='torch':
            result.update(torch_cuda=module.version.cuda,cxx11_abi=module._C._GLIBCXX_USE_CXX11_ABI)
    except Exception as error: result['imports'][name]=type(error).__name__+': '+str(error)
if result['versions']['flash-attn']:
    try:
        from flash_attn import flash_attn_varlen_func
        import flash_attn_2_cuda
        result['imports']['FA2']='ok'
    except Exception as error: result['imports']['FA2']=type(error).__name__+': '+str(error)
if result['versions']['flash-attn-4']:
    try:
        from flash_attn.cute.interface import flash_attn_varlen_func
        result['imports']['FA4']='ok'
    except Exception as error: result['imports']['FA4']=type(error).__name__+': '+str(error)
print(json.dumps(result),flush=True)
"""
for name in ('train','eval','train_fa2'):
    python=Path('/mnt/local/conda-py311/envs')/name/'bin/python3.11'
    if not python.is_file():
        print('RUNTIME_ENV '+json.dumps(dict(name=name,missing=str(python))),flush=True)
        continue
    try:
        result=subprocess.run([str(python),'-c',probe],capture_output=True,text=True,timeout=90)
        print('RUNTIME_ENV '+json.dumps(dict(name=name,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)),flush=True)
    except subprocess.TimeoutExpired:
        print('RUNTIME_ENV '+json.dumps(dict(name=name,timeout=True)),flush=True)
for args in (['--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu','--format=csv'],['--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv']):
    try:
        result=subprocess.run(['nvidia-smi',*args],capture_output=True,text=True,timeout=20)
        print('RUNTIME_GPU '+json.dumps(dict(args=args,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)),flush=True)
    except (subprocess.TimeoutExpired,FileNotFoundError) as error:
        print('RUNTIME_GPU '+json.dumps(dict(error=type(error).__name__)),flush=True)
print('RUNTIME_INSPECTION_COMPLETE',flush=True)
CHECK
