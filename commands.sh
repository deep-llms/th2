#1 +20+a
#th2-wc69-a2-ownership-check-20261103-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
export PYTHONDONTWRITEBYTECODE=1
python3 -u - <<'PY'
import json,os,hashlib,socket,subprocess,csv,io
from pathlib import Path
from datetime import datetime,timezone
def emit(k,v):print(k,json.dumps(v),flush=True)
emit('START',dict(utc=datetime.now(timezone.utc).isoformat(),host=socket.gethostname()))
r=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
rows=list(csv.reader(io.StringIO(r.stdout)));emit('GPU_COMPUTE_APPS',rows)
for pid in sorted({int(r[1].strip()) for r in rows if len(r)>1}):
    chain=[];cur=pid
    while cur>1 and len(chain)<12:
        try:
            p=Path('/proc')/str(cur);f=(p/'stat').read_text().rsplit(')',1)[1].split();cmd=(p/'cmdline').read_bytes();argv=[x.decode(errors='replace') for x in cmd.split(b'\0') if x];prefix=argv[:3]
            if '-c' in prefix:prefix=prefix[:prefix.index('-c')+1]
            chain.append(dict(pid=cur,ppid=int(f[1]),start_ticks=int(f[19]),exe=os.readlink(p/'exe'),prefix=prefix,cmd_sha256=hashlib.sha256(cmd).hexdigest()));cur=int(f[1])
        except OSError as e:chain.append(dict(error=str(e)));break
    emit('PROCESS_CHAIN',dict(worker=pid,chain=chain))
for s in ['/mnt/local','/mnt/local/deep-llms_th2/scripts','/mnt/local/_data','/mnt/local/_models','/mnt/local/_outputs']:
    p=Path(s);emit('DIRECTORY',dict(path=s,exists=p.exists(),children=sorted(x.name for x in p.iterdir()) if p.is_dir() else []))
for s in ['scripts/__init__.py','scripts/gpu_status.py','scripts/gpu_process.py','scripts/a2_depth_partc.py','/tmp/llm_pretrain_burn.py','/mnt/local/_gpu_guard/DISABLED']:
    p=Path(s);v=dict(path=s,exists=p.exists())
    if p.is_file():v.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    emit('FILE',v)
emit('COMPLETE',datetime.now(timezone.utc).isoformat())
PY
