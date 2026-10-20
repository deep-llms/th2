#1 +30+a
#th2-6y3o-fa2-final-verification-20261020-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
python3 - <<'WAIT'
from pathlib import Path
import time
work=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-fa2-build-20261019-a02')
started=time.monotonic()
while not any((work/name).is_file() for name in ('result.json','failure.json')):
    if time.monotonic()-started>1200:
        raise SystemExit('FA2 final verification timed out; inspect the existing build, do not relaunch')
    print('WAITING_FOR_EXISTING_FA2_BUILD',round(time.monotonic()-started),flush=True)
    time.sleep(30)
print('FA2_BUILD_TERMINAL_ARTIFACT_AVAILABLE',flush=True)
WAIT
python3 - <<'CHECK'
import datetime,json,os,socket,subprocess,shutil
from pathlib import Path

def emit(tag,obj): print(tag+' '+json.dumps(obj),flush=True)
emit('HEALTH_HOST',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),uptime=Path('/proc/uptime').read_text().strip(),loadavg=Path('/proc/loadavg').read_text().strip()))
mem={}
for line in Path('/proc/meminfo').read_text().splitlines():
    key,value=line.split(':',1)
    if key in ('MemTotal','MemAvailable','SwapTotal','SwapFree'): mem[key]=value.strip()
emit('HEALTH_MEMORY',mem)
for name in ('memory.events','memory.max','memory.current'):
    path=Path('/sys/fs/cgroup')/name
    if path.is_file(): emit('HEALTH_CGROUP',dict(name=name,value=path.read_text().strip()))
for path in ('/mnt/local','/tmp'):
    usage=shutil.disk_usage(path)
    emit('HEALTH_DISK',dict(path=path,total=usage.total,free=usage.free,used=usage.used))
probe=r"""
import importlib,json,sys
from importlib.metadata import version,PackageNotFoundError
out=dict(executable=sys.executable,versions={},imports={})
for name in ('torch','transformers','accelerate','datasets','tokenizers','safetensors','flash-attn-4','nvidia-cutlass-dsl','quack-kernels','lm-eval'):
    try: out['versions'][name]=version(name)
    except PackageNotFoundError: out['versions'][name]=None
for name in ('torch','transformers','accelerate','datasets','tokenizers','safetensors'):
    try:
        mod=importlib.import_module(name)
        out['imports'][name]='ok'
        if name=='torch': out.update(cuda=mod.version.cuda,cxx11abi=mod._C._GLIBCXX_USE_CXX11_ABI)
    except Exception as exc: out['imports'][name]=type(exc).__name__+': '+str(exc)
try:
    from flash_attn.cute.interface import flash_attn_varlen_func
    out['imports']['FA4']='ok'
except Exception as exc: out['imports']['FA4']=type(exc).__name__+': '+str(exc)
print(json.dumps(out),flush=True)
"""
for name in ('train','eval'):
    python='/mnt/local/conda-py311/envs/'+name+'/bin/python3.11'
    result=subprocess.run([python,'-c',probe],capture_output=True,text=True,timeout=90)
    emit('HEALTH_ENV',dict(name=name,rc=result.returncode,stdout=result.stdout,stderr=result.stderr))
for args in (['--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,ecc.errors.uncorrected.volatile.total','--format=csv'],['--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader']):
    result=subprocess.run(['nvidia-smi',*args],capture_output=True,text=True,timeout=25)
    emit('HEALTH_GPU',dict(args=args,rc=result.returncode,stdout=result.stdout,stderr=result.stderr))
    if result.returncode==0 and args[0].startswith('--query-compute-apps'):
        seen=set()
        for row in result.stdout.splitlines():
            pid=int(row.split(',')[0]); depth=0
            while pid>1 and pid not in seen and depth<4:
                seen.add(pid)
                path=Path('/proc')/str(pid)
                try:
                    raw=(path/'stat').read_text(); fields=raw.rsplit(')',1)[1].split()
                    ppid=int(fields[1]); start_ticks=int(fields[19])
                    args_raw=(path/'cmdline').read_bytes().split(b'\0')
                    scripts=[arg.decode(errors='replace') for arg in args_raw if arg.endswith(b'.py')]
                    emit('HEALTH_PROCESS',dict(pid=pid,ppid=ppid,start_ticks=start_ticks,exe=os.readlink(path/'exe'),scripts=scripts))
                    pid=ppid; depth+=1
                except (FileNotFoundError,PermissionError,ProcessLookupError): break
work=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-fa2-build-20261019-a02')
report_path=work/'result.json'
if not report_path.is_file():
    emit('FA2_BUILD_FAILED',json.loads((work/'failure.json').read_text()))
    raise SystemExit(1)
report=json.loads(report_path.read_text())
assert report['stage']=='complete'
emit('FA2_BUILD_RESULT',report)
probe="import json,torch,flash_attn,flash_attn_2_cuda; from flash_attn import flash_attn_func,flash_attn_varlen_func; print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,fa2=flash_attn.__version__,extension=flash_attn_2_cuda.__file__,symbols={name:hasattr(flash_attn_2_cuda,name) for name in ('fwd','bwd','varlen_fwd','varlen_bwd')})))"
clean_env=dict(os.environ)
for key in ('CUDA_HOME','CPATH','FLASH_ATTN_LOCAL_VERSION','FLASH_ATTENTION_FORCE_BUILD'):
    clean_env.pop(key,None)
result=subprocess.run(['/mnt/local/conda-py311/envs/train_fa2/bin/python3.11','-c',probe],env=clean_env,capture_output=True,text=True,timeout=90)
emit('FA2_CLEAN_IMPORT',dict(rc=result.returncode,stdout=result.stdout,stderr=result.stderr))
assert result.returncode==0
assert all(json.loads(result.stdout)['symbols'].values())
emit('HEALTH_CHECK_COMPLETE',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
CHECK
