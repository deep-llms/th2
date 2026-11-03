#1 +30+a
#th2-wc69-a2-runtime-assets-check-20261103-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true PYTORCH_NVML_BASED_CUDA_CHECK=1
python3 -u - <<'PY'
import json,os,socket,subprocess,sys,hashlib
from pathlib import Path
from datetime import datetime,timezone

def emit(label,value): print(label,json.dumps(value),flush=True)
def run(args,timeout=90):
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
        return dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
    except Exception as e:return dict(error=type(e).__name__+': '+str(e))
emit('CHECK_STARTED',dict(utc=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),cwd=os.getcwd()))
emit('GPU_DEVICES',run(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader']))
try:
    from scripts.gpu_status import snapshot
    from scripts.gpu_process import process
    q=subprocess.run(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],capture_output=True,text=True,check=True)
    devices=snapshot([int(x.strip()) for x in q.stdout.splitlines()])
    chains={}
    for pid in sorted({pid for g in devices for pid in g['pids']}):
        chain=[];cur=pid
        while cur>1 and len(chain)<12:
            try: row=process(cur)
            except (OSError,ValueError) as e:
                chain.append({'inspection_error':str(e)});break
            chain.append(row);cur=row['ppid']
        chains[str(pid)]=chain
    emit('GPU_OWNERSHIP',dict(devices=devices,chains=chains))
except Exception as e:emit('GPU_INSPECTION_ERROR',str(e))
probe='''import sys,json,importlib,importlib.metadata as m
r={'executable':sys.executable,'python':sys.version,'versions':{},'imports':{}}
for name in ['torch','transformers','accelerate','datasets','numpy','safetensors','tokenizers','huggingface-hub','lm_eval']:
 try:r['versions'][name]=m.version(name)
 except Exception:r['versions'][name]=None
for name in ['torch','transformers','accelerate','datasets','numpy','safetensors']+(['lm_eval'] if sys.argv[1]=='eval' else []):
 try:importlib.import_module(name);r['imports'][name]='OK'
 except Exception as e:r['imports'][name]=type(e).__name__+': '+str(e)
try:
 import torch
 r['cuda_build']=torch.version.cuda;r['cuda_available_nvml']=torch.cuda.is_available();r['cuda_count_nvml']=torch.cuda.device_count();r['cuda_initialized']=torch.cuda.is_initialized()
except Exception as e:r['cuda_error']=str(e)
try:
 from transformers import Trainer,AutoModelForCausalLM,AutoTokenizer
 r['trainer_import']='OK'
except Exception as e:r['trainer_import']=type(e).__name__+': '+str(e)
print(json.dumps(r))
'''
for env in ('a2_env','eval','train_env'):
    p=Path('/mnt/local/conda-py311/envs')/env/'bin/python3.11'
    emit('ENV_PATH',dict(env=env,path=str(p),exists=p.is_file()))
    if p.is_file():
        emit('ENV_PROBE',dict(env=env,result=run([str(p),'-B','-c',probe,env])))
        emit('PIP_CHECK',dict(env=env,result=run([str(p),'-B','-m','pip','check'])))
for base in ('/mnt/local/_data/deep-llms_th2','/mnt/local/_models/deep-llms_th2','/mnt/local/_outputs/deep-llms_th2','/mnt/local/conda-py311/envs'):
    p=Path(base);emit('ROOT_INVENTORY',dict(path=base,exists=p.exists(),children=sorted(x.name for x in p.iterdir()) if p.is_dir() else []))
plan=json.loads(Path('resources/a2_depth_partc_plan_20261101.json').read_text());paths=set()
def collect(x):
    if isinstance(x,dict):
        for v in x.values():collect(v)
    elif isinstance(x,list):
        for v in x:collect(v)
    elif isinstance(x,str) and x.startswith('/mnt/local/') and '\n' not in x:paths.add(x)
collect(plan)
for s in sorted(paths):
    p=Path(s);emit('REGISTERED_PATH',dict(path=s,exists=p.exists(),is_dir=p.is_dir(),bytes=p.stat().st_size if p.is_file() else None))
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-depth-partc-20261101-a02')
for rel in ('source-manifest.json','supervised/supervisor.json','supervised/run/complete.json','results/a2_depth_20261031/progress.json','results/a2_depth_20261031/attempts.json','results/a2_depth_20261031/downstream/complete.json'):
    p=root/rel
    if p.is_file() and p.stat().st_size<1000000:
        raw=p.read_bytes();emit('ARTIFACT',dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode()))
    else:emit('ARTIFACT_UNAVAILABLE',str(p))
for p in sorted(root.glob('results/a2_depth_20261031/heals/*/trainer/**/depth-checkpoint.json')):
    try:
        marker=json.loads(p.read_text());actual={n:((p.parent/n).stat().st_size if (p.parent/n).is_file() else None) for n in marker['files']}
        emit('CHECKPOINT_MARKER',dict(path=str(p),marker=marker,actual_sizes=actual,sizes_match=actual==marker['files']))
    except Exception as e:emit('CHECKPOINT_INSPECTION_ERROR',dict(path=str(p),error=str(e)))
emit('DISK',run(['df','-h','/mnt/local']))
emit('CHECK_COMPLETE',dict(utc=datetime.now(timezone.utc).isoformat(),note='Inspection only; no GPU tensors, resume, training, signals, installs or downloads.'))
PY
