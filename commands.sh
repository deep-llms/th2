#1 +20+a
#th2-wc69-a2-why-fwe-locate-20261104-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PYREMOTE'
import os,json,socket
from pathlib import Path
from datetime import datetime,timezone
root=Path('/mnt/local/_data')
targets={'000_00007.parquet','002_00002.parquet','001_00004.parquet'}
r=dict(utc=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),scanned_root=str(root),visited=0,matches=[],fwe_directories=[],errors=[],truncated=False)
for base,dirs,files in os.walk(root,followlinks=False,onerror=lambda e:r['errors'].append(str(e))):
 r['visited']+=1
 if r['visited']>20000:r['truncated']=True;break
 if 'fineweb' in base.lower():r['fwe_directories'].append(dict(path=base,files=len(files),subdirs=len(dirs)))
 for n in files:
  if n in targets:
   p=Path(base)/n;r['matches'].append(dict(path=str(p),bytes=p.stat().st_size,symlink=p.is_symlink()))
for n in ['/mnt/local','/mnt/local/_data','/mnt/local/_data/deep-llms_th2','/mnt/local/_data/deep-llms_th2/a2_corpora_30b_20261029/raw/fineweb_edu']:
 p=Path(n);st=p.stat();r.setdefault('directories',[]).append(dict(path=n,device=st.st_dev,inode=st.st_ino,uid=st.st_uid,gid=st.st_gid,mode=oct(st.st_mode),resolved=str(p.resolve())))
print('FWE_LOCATION_JSON',json.dumps(r),flush=True)
PYREMOTE
