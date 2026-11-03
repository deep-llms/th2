#1 +20+a
#th2-wc69-a2-backup-import-check-20261103-a03
set -euo pipefail
cd /mnt/local/deep-llms_th2
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
python3 -u - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,socket
print('START',json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),host=socket.gethostname())),flush=True)
root=Path('/mnt/local/_backup')
queue=[(root,0)];count=0
while queue and count<250:
    p,depth=queue.pop(0)
    print('BACKUP_ENTRY',json.dumps(dict(path=str(p),directory=p.is_dir(),symlink=p.is_symlink(),bytes=p.stat().st_size if p.is_file() else None)),flush=True);count+=1
    if p.is_dir() and not p.is_symlink() and depth<2:queue.extend((c,depth+1) for c in sorted(p.iterdir()))
print('BACKUP_LIST_LIMIT',json.dumps(dict(entries=count,truncated=bool(queue),max_depth=2)),flush=True)
probe='''import sys,json,importlib
r={'python':sys.executable,'sys_path':sys.path,'modules':{}}
for name in ['scripts','scripts.gpu_status','scripts.gpu_process']:
 try:
  m=importlib.import_module(name);r['modules'][name]={'file':getattr(m,'__file__',None),'path':list(getattr(m,'__path__',[]))}
 except Exception as e:r['modules'][name]={'error':type(e).__name__+': '+str(e)}
print(json.dumps(r))'''
for exe in ['/usr/bin/python3','/mnt/local/conda-py311/envs/a2_env/bin/python3.11']:
    r=subprocess.run([exe,'-B','-c',probe],capture_output=True,text=True,timeout=30)
    print('IMPORT_ORIGIN',json.dumps(dict(exe=exe,rc=r.returncode,stdout=r.stdout,stderr=r.stderr)),flush=True)
print('COMPLETE',datetime.now(timezone.utc).isoformat(),flush=True)
PY
