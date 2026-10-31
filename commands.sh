#1 +30+a
#th2-6y3o-node-identity-20261031-a01
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
python3 -u - <<'PY'
import hashlib,json,os,socket,subprocess,time
from pathlib import Path
from datetime import datetime,timezone

def emit(tag,value):
 print(tag,json.dumps(value,ensure_ascii=False),flush=True)
def read(path,limit=5000):
 try:
  with Path(path).open('rb') as f: return f.read(limit).decode(errors='replace')
 except Exception as e: return {'error':type(e).__name__,'detail':str(e)}
def command(args):
 try:
  p=subprocess.run(args,capture_output=True,text=True,timeout=15)
  return {'returncode':p.returncode,'stdout':p.stdout[:14000],'stderr':p.stderr[:1000]}
 except Exception as e:return {'error':type(e).__name__,'detail':str(e)}
emit('IDENTITY_BEGIN',{'utc':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),'cwd':os.getcwd(),'uname':list(os.uname())})
for path in ('/proc/sys/kernel/random/boot_id','/proc/uptime','/sys/class/dmi/id/product_uuid','/sys/class/dmi/id/product_name'):
 emit('HOST_FILE',{'path':path,'value':read(path,1000)})
uptime=read('/proc/uptime',1000)
if isinstance(uptime,str):
 try:emit('BOOT_UTC_ESTIMATE',datetime.fromtimestamp(time.time()-float(uptime.split()[0]),timezone.utc).isoformat())
 except Exception as e:emit('BOOT_ERROR',str(e))
emit('GPU_IDENTITY',command(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']))
emit('GPU_COMPUTE',command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']))
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01')
base=root/'results/a2_quant_20261031'
paths=[Path('/mnt/local'),Path('/mnt/local/_outputs/deep-llms_th2'),Path('/mnt/local/_data/deep-llms_th2'),Path('/mnt/local/_models/deep-llms_th2'),Path('/mnt/local/conda-py311/envs/a2_env/bin/python3.11'),root,root/'source',base]
for p in paths:
 try:
  s=p.stat();item={'path':str(p),'exists':True,'mtime_utc':datetime.fromtimestamp(s.st_mtime,timezone.utc).isoformat(),'inode':s.st_ino,'device':s.st_dev}
  if p.is_dir():item['entries_first30']=sorted(x.name for x in p.iterdir())[:30]
 except Exception as e:item={'path':str(p),'exists':False,'error':type(e).__name__,'detail':str(e)}
 emit('PATH_STATE',item)
for rel in ('source-manifest.json','supervised/supervisor.json','supervised/run/complete.json','supervised/run/quant-complete.json','results/a2_quant_20261031/complete.json','results/a2_quant_20261031/bitwidth.json'):
 p=root/rel
 if p.is_file():emit('SMALL_ARTIFACT',{'path':str(p),'bytes':p.stat().st_size,'first5000':read(p)})
 else:emit('MISSING',str(p))
for pid in (736203,736243,736903,736904,736907,736910,736911,736912,736913,736914):
 p=Path('/proc')/str(pid)
 emit('KNOWN_PID',{'pid':pid,'exists':p.exists(),'stat':read(p/'stat',2000) if p.exists() else None,'comm':read(p/'comm',200) if p.exists() else None})
for pattern in ('workers/*/*/*/complete.json','workers/*/*/*/*-final.json','workers/*/*/*/*-own_corpus.json'):
 emit('ARTIFACT_COUNT',{'pattern':pattern,'count':len(list(base.glob(pattern)))})
for rel in ('supervisor.log','supervised/run/a2-quant.log','supervised/burn.log'):
 p=root/rel
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-1500));tail=f.read(1500).decode(errors='replace')
  emit('LOG_TAIL',{'path':str(p),'bytes':p.stat().st_size,'text':tail})
emit('QUANT_SESSION',command(['tmux','display-message','-p','-t','a2_quant_20261031_a01','#{pane_dead} #{pane_pid}']))
emit('IDENTITY_COMPLETE',datetime.now(timezone.utc).isoformat())
PY
