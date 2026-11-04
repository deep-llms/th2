#1 +20+a
#th2-wc69-a2-why-fwe-batchverify-20261105-b01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PYREMOTE'
import os,json,hashlib,socket,subprocess
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect
raw=Path('/mnt/local/_data/deep-llms_th2/a2_corpora_30b_20261029/raw/fineweb_edu')
r=dict(utc=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),gpu=inspect(),raw_path=str(raw),raw_exists=raw.exists(),raw_symlink=raw.is_symlink(),raw_resolved=str(raw.resolve()),files=[],directories=[],processes=[],unreadable_pids=[])
if raw.exists():
 for p in sorted(raw.rglob('*')):
  if p.is_symlink():r.setdefault('symlinks',[]).append(str(p));continue
  if p.is_file():r['files'].append(dict(path=str(p),bytes=p.stat().st_size,mtime=p.stat().st_mtime))
  elif p.is_dir():r['directories'].append(str(p))
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cmd=(p/'cmdline').read_bytes();args=cmd.decode(errors='replace').split('\0');stat=(p/'stat').read_text().rsplit(')',1)[1].split();comm=(p/'comm').read_text().strip()
  words=[Path(a).name for a in args[:3]]
  matched=any(x in cmd.lower() for x in [b'fineweb',b'a2_why_prepare',b'a2_prepare_corpora']) or any(x in words for x in ['hf','huggingface-cli','wget','curl','aria2c','rsync','scp','rclone','aws','s5cmd'])
  fds=[]
  for fd in (p/'fd').iterdir():
   try:target=os.readlink(fd)
   except OSError:continue
   if target.startswith(str(raw)+'/'):fds.append(target)
  if matched or fds:
   r['processes'].append(dict(pid=int(p.name),ppid=int(stat[1]),state=stat[0],start_ticks=int(stat[19]),comm=comm,exe=os.readlink(p/'exe'),command_sha256=hashlib.sha256(cmd).hexdigest(),is_preparer=b'a2_why_prepare' in cmd or b'a2_prepare_corpora' in cmd,raw_open_fds=fds,io=(p/'io').read_text()))
 except (FileNotFoundError,ProcessLookupError):continue
 except PermissionError:r['unreadable_pids'].append(int(p.name))

items=[{'index': 0, 'path': 'sample/100BT/000_00007.parquet', 'bytes': 2150919214, 'sha256': '98e15e6976fe0e9cafe4e695d37ca836cae9ffa3c4c3937f8f1325dc6bf899e8', 'local': '/mnt/local/_data/deep-llms_th2/a2_corpora_30b_20261029/raw/fineweb_edu/00000.parquet', 'url': 'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/100BT/000_00007.parquet'}, {'index': 1, 'path': 'sample/100BT/002_00002.parquet', 'bytes': 2151150989, 'sha256': '65b07ff5f637f3ad2ac7f3cedc00c4e146e403f777f04d9854c8b981d15400f7', 'local': '/mnt/local/_data/deep-llms_th2/a2_corpora_30b_20261029/raw/fineweb_edu/00001.parquet', 'url': 'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/100BT/002_00002.parquet'}]
r['batch_files']=[]
for i in items:
 p=Path(i['local'])/Path(i['path']).name
 q=dict(path=str(p),expected_bytes=i['bytes'],expected_sha256=i['sha256'],exists=p.is_file(),pass_size=False,pass_sha256=False)
 if p.is_file():
  q['bytes']=p.stat().st_size;q['pass_size']=q['bytes']==i['bytes']
  if q['pass_size']:
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
   q['sha256']=h.hexdigest();q['pass_sha256']=q['sha256']==i['sha256']
 r['batch_files'].append(q)

print('FWE_INSPECTION_JSON',json.dumps(r),flush=True)
PYREMOTE
