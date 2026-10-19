#1 +30+a
#th2-6y3o-fa2-build-preflight-20261019-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
python3 - <<'CHECK'
import datetime,json,os,socket,subprocess,shutil
from pathlib import Path
print('BUILD_PREFLIGHT_HOST '+json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname())),flush=True)
code=r"""
import json,sys,os,shutil
from importlib.metadata import version,PackageNotFoundError
import torch
from torch.utils.cpp_extension import CUDA_HOME
out=dict(executable=sys.executable,prefix=sys.prefix,torch=torch.__version__,cuda=torch.version.cuda,abi=torch._C._GLIBCXX_USE_CXX11_ABI,cuda_home=CUDA_HOME,versions={},tools={})
for name in ('ninja','packaging','wheel','setuptools','psutil','pip','einops','cuda-toolkit','nvidia-cuda-nvcc','nvidia-cuda-nvcc-cu13','nvidia-cuda-runtime','nvidia-cuda-runtime-cu13','flash-attn'):
    try: out['versions'][name]=version(name)
    except PackageNotFoundError: out['versions'][name]=None
for name in ('nvcc','g++','gcc','ninja','cmake','timeout','prlimit','nice'):
    out['tools'][name]=shutil.which(name)
print(json.dumps(out),flush=True)
"""
python='/mnt/local/conda-py311/envs/train_fa2/bin/python3.11'
env=dict(os.environ); env['PATH']=str(Path(python).parent)+':'+env['PATH']
r=subprocess.run([python,'-c',code],env=env,capture_output=True,text=True,timeout=60)
print('BUILD_PREFLIGHT_ENV '+json.dumps(dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr)),flush=True)
candidates=set(Path('/usr/local').glob('cuda*/bin/nvcc'))
found=shutil.which('nvcc')
if found: candidates.add(Path(found))
for nvcc in sorted(candidates):
    r=subprocess.run([str(nvcc),'--version'],capture_output=True,text=True,timeout=15)
    print('BUILD_PREFLIGHT_NVCC '+json.dumps(dict(path=str(nvcc),realpath=str(nvcc.resolve()),rc=r.returncode,stdout=r.stdout,stderr=r.stderr)),flush=True)
for tool in ('g++','gcc'):
    if shutil.which(tool):
        r=subprocess.run([tool,'--version'],capture_output=True,text=True,timeout=15)
        print('BUILD_PREFLIGHT_COMPILER '+json.dumps(dict(tool=tool,rc=r.returncode,version=r.stdout.splitlines()[0])),flush=True)
for path in ('/mnt/local','/tmp'):
    stat=shutil.disk_usage(path)
    print('BUILD_PREFLIGHT_DISK '+json.dumps(dict(path=path,free=stat.free)),flush=True)
print('BUILD_PREFLIGHT_COMPLETE',flush=True)
CHECK
