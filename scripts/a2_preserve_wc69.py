"""Additive project backup; all file copies verified, originals never replaced."""
import os,json,hashlib,shutil,time,socket,stat
from pathlib import Path
from datetime import datetime,timezone
B=Path('/mnt/local/_backup/deep-llms_th2')
O=Path('/mnt/local/_outputs/deep-llms_th2')
A=B/'wc69-preservation-20261111-a01'
assert socket.gethostname()=='thiennh-p6-wc69-worker-0'
assert B.is_dir() and O.is_dir() and not B.is_symlink()
A.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(name,v):
 p=A/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def stamp():return datetime.now(timezone.utc).isoformat()
files=[];excluded=[]
for root,label in [(Path('/mnt/local/deep-llms_th2'),'source'),(O,'outputs'),(Path('/mnt/local/_models/deep-llms_th2'),'models'),(Path('/mnt/local/_data/deep-llms_th2'),'data_metadata')]:
 assert root.is_dir()
 for d,dirs,names in os.walk(root,followlinks=False):
  for n in list(dirs):
   p=Path(d)/n
   if p.is_symlink():raise RuntimeError('Directory symlink: '+str(p))
   if n.startswith('checkpoint-') or n in ('__pycache__','.git','wc69-preservation-control-20261111-a01'):
    dirs.remove(n);excluded.append(str(p))
  for n in sorted(names):
   p=Path(d)/n;rel=p.relative_to(root)
   if label=='data_metadata' and not (n.endswith(('.json','.jsonl','.sha256')) or any(k in n.lower() for k in ('order','permutation','indices'))):continue
   if p.is_symlink():raise RuntimeError('File symlink: '+str(p))
   if not stat.S_ISREG(p.stat().st_mode):raise RuntimeError('Nonregular file: '+str(p))
   if n in ('optimizer.pt','scheduler.pt') or n.startswith('rng_state') or n.endswith('.pyc'):
    excluded.append(str(p));continue
   if any('burn' in x for x in rel.parts) and n.endswith(('.log','.jsonl')):
    excluded.append(str(p));continue
   dst=(A/label/rel) if label in ('source','data_metadata') else B/rel
   if dst.is_symlink():raise RuntimeError('Target symlink: '+str(dst))
   files.append((p,dst,label,rel,p.stat().st_size))
# Preflight uses a conservative bound for every absent or different-size target.
need=sum(s for p,t,l,r,s in files if not t.exists() or t.stat().st_size!=s)
free=shutil.disk_usage(B).free
save('plan.json',dict(utc=stamp(),files=len(files),candidate_bytes=sum(x[4] for x in files),missing_or_size_different_bytes=need,free_bytes=free,excluded=excluded))
assert free>need+(10<<30),'Insufficient free space for backup'
print('BACKUP_PLAN',json.dumps(dict(files=len(files),bytes=sum(x[4] for x in files),estimated_new_bytes=need,free_bytes=free)),flush=True)
stats=dict(copied=0,reused=0,conflicts_preserved=0,copied_bytes=0)
with (A/'files.jsonl').open('x') as ledger:
 for i,(p,t,label,rel,s) in enumerate(files):
  before=p.stat();h=sha(p)
  if t.is_file() and t.stat().st_size==s and sha(t)==h:
   stats['reused']+=1;action='reused'
  else:
   if t.exists():
    t=A/'conflicts'/label/rel;stats['conflicts_preserved']+=1
   assert not t.exists(),'Destination unexpectedly exists'
   assert shutil.disk_usage(B).free>s+(2<<30),'Insufficient free space'
   # Refuse writing through any existing destination directory symlink.
   for ancestor in t.parents:
    if ancestor==B:break
    assert not ancestor.is_symlink(),'Destination ancestor symlink'
   t.parent.mkdir(parents=True,exist_ok=True)
   tmp=t.with_name(t.name+'.backup-part-a01');assert not tmp.exists()
   with p.open('rb') as src,tmp.open('xb') as dst:
    shutil.copyfileobj(src,dst,8<<20);dst.flush();os.fsync(dst.fileno())
   assert tmp.stat().st_size==s and sha(tmp)==h,'Copy hash mismatch'
   after=p.stat()
   assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Source changed during backup'
   os.link(tmp,t);tmp.unlink()
   stats['copied']+=1;stats['copied_bytes']+=s;action='copied'
  ledger.write(json.dumps(dict(source=str(p),destination=str(t),bytes=s,sha256=h,action=action))+'\n');ledger.flush()
  if i%100==0:
   save('progress.json',dict(utc=stamp(),processed=i+1,total=len(files),**stats));print('BACKUP_PROGRESS',i+1,len(files),stats,flush=True)
# A fresh destination pass ensures the entire ledger remains readable and matches.
for line in (A/'files.jsonl').read_text().splitlines():
 v=json.loads(line);p=Path(v['destination']);assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256']
save('complete.json',dict(complete=True,utc=stamp(),files=len(files),manifest_sha256=sha(A/'files.jsonl'),destination=str(B),**stats))
print('BACKUP_COMPLETE',json.dumps(json.loads((A/'complete.json').read_text())),flush=True)
