#1 +20+a
#th2-6y3o-project-backup-inventory-20261031-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
python3 -u - <<'PYCODE'
import os,json,socket,subprocess
from pathlib import Path
from datetime import datetime,timezone
print('BACKUP_INVENTORY_START',datetime.now(timezone.utc).isoformat(),socket.gethostname(),flush=True)
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
roots=['/mnt/local/_outputs/deep-llms_th2','/mnt/local/_models/deep-llms_th2','/mnt/local/_data/deep-llms_th2','/mnt/local/deep-llms_th2']
for root in roots:
 r=subprocess.run(['du','-B1','--max-depth=1',root],capture_output=True,text=True)
 print('DU_ALLOCATED',root,r.returncode,flush=True)
 print(r.stdout,flush=True)
 if r.stderr: print('DU_ERROR',r.stderr,flush=True)
 groups={}; links=[]
 for directory,dirs,files in os.walk(root,followlinks=False):
  for name in files:
   p=Path(directory)/name
   if p.is_symlink():
    links.append([str(p),os.readlink(p)]);continue
   s=p.stat(); rel=p.relative_to(root); group=rel.parts[0] if len(rel.parts)>1 else '.'
   a=groups.setdefault(group,{'files':0,'bytes':0,'weights_bytes':0,'optimizer_bytes':0,'weight_files':0})
   a['files']+=1;a['bytes']+=s.st_size
   if 'optimizer' in name.lower(): a['optimizer_bytes']+=s.st_size
   elif p.suffix in ('.safetensors','.pt','.pth') or name=='pytorch_model.bin':
    a['weights_bytes']+=s.st_size;a['weight_files']+=1
 print('LOGICAL_INVENTORY',root,json.dumps(groups,sort_keys=True),flush=True)
 print('SYMLINKS',root,json.dumps(links[:100]),'total',len(links),flush=True)
print('BACKUP_INVENTORY_COMPLETE',datetime.now(timezone.utc).isoformat(),flush=True)
PYCODE
