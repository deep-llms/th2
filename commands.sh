#1 +30+a
#th2-wc69-dot-pool-inventory-20261112-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python3 - <<'PY'
import json,os,socket,subprocess,hashlib
from pathlib import Path
from datetime import datetime,timezone
def cmd(a):
 p=subprocess.run(a,capture_output=True,text=True);return dict(code=p.returncode,out=p.stdout[-24000:],err=p.stderr[-1000:])
x={'utc':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),'head':cmd(['git','rev-parse','HEAD']),'gpu':cmd(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader']),'compute':cmd(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'])}
pids=[int(z.split(',')[0]) for z in x['compute']['out'].splitlines() if z.split(',')[0].strip().isdigit()]
x['processes']={}
for pid in pids:
 for _ in range(6):
  if pid in x['processes'] or pid<2: break
  try:
   p=Path('/proc')/str(pid);st=p.joinpath('stat').read_text();parts=st[st.rfind(')')+2:].split();x['processes'][pid]={'cmd':p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(),'starttime':parts[19],'ppid':int(parts[1])};pid=int(parts[1])
  except FileNotFoundError:break
x['envs']={}
for e in ['a2_env','train_env','eval']:
 p=Path('/mnt/local/conda-py311/envs')/e/'bin/python3.11'
 if p.exists():x['envs'][e]=cmd([str(p),'-c',"import torch,transformers,importlib.util,json; print(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,grouped=hasattr(torch.nn.functional,'grouped_mm'),fa4=importlib.util.find_spec('flash_attn') is not None)))"])
x['paths']={}
for base in ['/mnt/local/_data/deep-llms_th2','/mnt/local/_outputs/deep-llms_th2/wc69-a2-data-rebuild-20261103-a02','/mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs/data','/mnt/local/_models/deep-llms_th2/a2/dclm/P300']:
 p=Path(base);x['paths'][base]={'exists':p.exists(),'children':[str(c) for c in p.glob('*')][:50]}
 if p.is_dir():
  x['paths'][base]['metadata']={str(f.relative_to(p)):json.loads(f.read_text()) for name in ['manifest.json','complete.json','tokenizer_config.json'] if (f:=p/name).exists() and f.stat().st_size<120000}
x['disk']=cmd(['df','-h','/mnt/local'])
print('DOT_INVENTORY_JSON='+json.dumps(x))
PY
