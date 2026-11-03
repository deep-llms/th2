#1 +30+a
#th2-wc69-depth-backup-audit-20261103-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import os,json,hashlib,socket
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
root=Path('/mnt/local/_backup');totals={};matches=[];states=[];errors=[];links=[];digest=hashlib.sha256();dirs=0
print('AUDIT_START',json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),root=str(root))),flush=True)
def err(e):errors.append(str(e))
for base,sub,files in os.walk(root,followlinks=False,onerror=err):
    sub.sort();files.sort();dirs+=1
    for name in sub+files:
        p=Path(base)/name;rel=p.relative_to(root);parts=rel.parts;top='/'.join(parts[:2])
        if p.is_symlink():links.append(dict(path=str(rel),target=os.readlink(p)));continue
        if name not in files:continue
        try:s=p.stat()
        except OSError as e:err(e);continue
        v=totals.setdefault(top,dict(files=0,bytes=0,weights=0,optimizer=0,rng=0,oldest_mtime=None,newest_mtime=None))
        v['files']+=1;v['bytes']+=s.st_size
        v['weights']+=int(name.endswith('.safetensors') or name=='pytorch_model.bin')
        v['optimizer']+=int(name=='optimizer.pt');v['rng']+=int(name.startswith('rng_state'))
        v['oldest_mtime']=min(v['oldest_mtime'] or s.st_mtime,s.st_mtime);v['newest_mtime']=max(v['newest_mtime'] or s.st_mtime,s.st_mtime)
        digest.update(json.dumps([str(rel),s.st_size,s.st_mtime_ns]).encode()+b'\n')
        if 'depth' in str(rel).lower() or '20261101' in str(rel) or '20261102' in str(rel):
            matches.append(dict(path=str(rel),bytes=s.st_size))
            if len(matches)<=100 and s.st_size<65536 and p.suffix in ('.json','.md'):
                raw=p.read_bytes();print('MATCH_CONTENT',json.dumps(dict(path=str(rel),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode(errors='replace'))),flush=True)
        if name=='trainer_state.json':
            try:
                x=json.loads(p.read_text());states.append(dict(path=str(rel),global_step=x.get('global_step'),max_steps=x.get('max_steps')))
            except Exception as e:err(e)
for top,v in sorted(totals.items()):print('BACKUP_TOTAL',json.dumps(dict(root=top,**v)),flush=True)
print('DEPTH_CANDIDATES',json.dumps(dict(count=len(matches),entries=matches[:300],truncated=len(matches)>300)),flush=True)
print('TRAINER_STATES',json.dumps(dict(count=len(states),entries=states[:300],truncated=len(states)>300)),flush=True)
print('SYMLINKS',json.dumps(dict(count=len(links),entries=links[:100],truncated=len(links)>100)),flush=True)
print('AUDIT_COMPLETE',json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),directories=dirs,files=sum(x['files'] for x in totals.values()),bytes=sum(x['bytes'] for x in totals.values()),inventory_sha256=digest.hexdigest(),errors=errors,root_exists=root.is_dir(),note='Recursive metadata audit; no symlink traversal and no large weight hashes.')),flush=True)
PY
