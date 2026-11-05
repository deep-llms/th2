#1 +120+a
#th2-wc69-a2-why-fwe-rawaudit-20261105-a10
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

items=json.loads(Path('resources/a2_corpora_30b_20261029.json').read_text())['corpora']['fineweb_edu']['files'][:44]
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


r['preparation']={}
for root,names in [('/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-data-20261104-a01',['complete.json','failure.json','historical_fwe_identity.json','tests.json']),('/mnt/local/_data/deep-llms_th2/a2_why_fwe_rebuild_20261104',['status.json','complete.json'])]:
 for name in names:
  p=Path(root)/name
  if p.is_file():r['preparation'][str(p)]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),value=json.loads(p.read_text()))

r['tokenized_parts']=[]
partroot=Path('/mnt/local/_data/deep-llms_th2/a2_why_fwe_rebuild_20261104/parts/fineweb_edu')
for item in items:
 receipt=partroot/f"{item['index']:05d}"/'complete.json'
 if receipt.is_file():
  v=json.loads(receipt.read_text())
  r['tokenized_parts'].append(dict(index=item['index'],status=v.get('status'),source_matches=v.get('source')==item,receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),rows=v.get('rows'),streams=v.get('streams')))


r['partial_parts']=[]
if partroot.exists():
 for d in sorted(partroot.iterdir()):
  if d.is_dir():
   r['partial_parts'].append(dict(index=d.name,complete=(d/'complete.json').is_file(),files=[dict(name=p.name,bytes=p.stat().st_size,mtime=p.stat().st_mtime) for p in sorted(d.iterdir()) if p.is_file()]))


r['disk_usage']={}
import shutil
for path in [raw,partroot]:
 u=shutil.disk_usage(path);r['disk_usage'][str(path)]=dict(total=u.total,used=u.used,free=u.free)
prep_root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-data-20261104-a01')
manifest=json.loads((prep_root/'source-manifest.json').read_text())
r['preparation_source_unchanged']=all(hashlib.sha256((prep_root/'source'/n).read_bytes()).hexdigest()==h for n,h in manifest['files'].items())
pause_root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-pause-20261104-a01/pause')
saved=json.loads((pause_root/'saved_state.json').read_text())
r['pause_checkpoint']={'global_step':saved['global_step'],'path':saved['checkpoint'],'files':[]}
for name,expected in saved['hashes'].items():
 p=Path(saved['checkpoint'])/name;h=hashlib.sha256()
 if p.is_file():
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 r['pause_checkpoint']['files'].append(dict(name=name,exists=p.is_file(),expected_sha256=expected,sha256=h.hexdigest() if p.is_file() else None,passed=p.is_file() and h.hexdigest()==expected))

r['checked_end_utc']=datetime.now(timezone.utc).isoformat()

print('FWE_INSPECTION_JSON',json.dumps(r),flush=True)
PYREMOTE
