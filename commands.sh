#1 +30+a
#th2-6y3o-a2-corpora-health-20261029-a09
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
from pathlib import Path
import json,hashlib,socket,subprocess
from datetime import datetime,timezone
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-corpora-20261029-a03')
data=Path('/mnt/local/_data/deep-llms_th2/a2_corpora_30b_20261029')
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
for label,p in [('source-manifest.json',root/'source-manifest.json'),('data/status.json',data/'status.json'),('data/failure.json',data/'failure.json'),('data/complete.json',data/'complete.json'),*[(f'data/{c}/manifest.json',data/'packed'/c/'manifest.json') for c in ('dclm','fineweb_edu')]]:
 if p.is_file():
  b=p.read_bytes();print('ARTIFACT',json.dumps(dict(path=label,sha256=hashlib.sha256(b).hexdigest(),text=b.decode())),flush=True)
 else:print('MISSING',label,flush=True)
plan=json.loads((root/'source/plan.json').read_text())
for name,corpus in plan['corpora'].items():
 files=[Path(f['local'])/Path(f['path']).name if Path(f['local']).is_dir() else Path(f['local']) for f in corpus['files']];present=[p for p in files if p.is_file()]
 parts=list((data/'parts'/name).glob('*/complete.json'))
 print('CORPUS_COUNTS',json.dumps(dict(corpus=name,raw_files_present=len(present),raw_bytes_observed=sum(p.stat().st_size for p in present),planned_files=len(files),completed_parts=len(parts))),flush=True)
log=root/'prepare.log'
if log.is_file():
 with log.open('rb') as f:f.seek(max(0,log.stat().st_size-12000));tail=f.read().decode(errors='replace')
 print('LOG_TAIL',json.dumps(dict(path='prepare.log',bytes=log.stat().st_size,text=tail)),flush=True)
print('SESSION',subprocess.run(['tmux','display-message','-p','-t','a2_corpora_20261029_a03','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip(),flush=True)
raw=data/'raw'
if raw.exists():
 for name in ('dclm','fineweb_edu'):
  base=raw/name
  entries=[]
  if base.exists():
   for p in sorted(base.iterdir())[:12]:
    entries.append(dict(name=p.name,is_file=p.is_file(),is_dir=p.is_dir(),bytes=p.stat().st_size,children=[dict(name=q.name,is_file=q.is_file(),bytes=q.stat().st_size) for q in sorted(p.iterdir())[:2]] if p.is_dir() else []))
  print('RAW_LAYOUT',json.dumps(dict(corpus=name,entries=entries)),flush=True)
processes=subprocess.run(['ps','-eo','pid,ppid,pcpu,pmem,etime,args'],capture_output=True,text=True,check=True).stdout.splitlines()
parents={x.split()[0] for x in processes if str(root/'source/scripts/a2_prepare_corpora.py') in x}; selected=[x for x in processes if len(x.split())>2 and (x.split()[0] in parents or x.split()[1] in parents)]
print('CPU_PROCESSES',json.dumps(selected),flush=True)
for name,corpus in plan['corpora'].items():
 base=data/'parts'/name
 receipts=[]
 if base.exists():
  for p in sorted(base.glob('*/complete.json')):
   item=json.loads(p.read_text());record=corpus['files'][item['source']['index']]
   assert item['source']==record
   assert item['tokenizer_sha256']==plan['tokenizer_files']['tokenizer.json']
   receipts.append(dict(index=item['source']['index'],seconds=item['seconds'],tokens=sum(x['tokens'] for x in item['streams'].values()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 token_bytes=sum(p.stat().st_size for p in base.glob('*/*.bin') if p.name!='documents.bin') if base.exists() else 0
 print('PART_PROGRESS',json.dumps(dict(corpus=name,completed=receipts,observed_token_bytes=token_bytes,observed_tokens=token_bytes//4)),flush=True)
full=data/'packed/dclm'
snap=data/'snapshots/dclm_first_3p2b_20261029'
if (full/'manifest.json').is_file() and (snap/'complete.json').is_file():
 fm=json.loads((full/'manifest.json').read_text());sm=json.loads((snap/'manifest.json').read_text())
 assert fm['status']=='complete' and fm['plan_sha256']==sm['plan_sha256']
 results={}
 for name,mult,key in [('heal.bin',4,'token_sha256'),('heal.starts',1,'starts_sha256')]:
  remaining=sm['streams']['heal']['tokens']*mult;h=hashlib.sha256()
  with (full/name).open('rb') as f:
   while remaining:
    b=f.read(min(8<<20,remaining));assert b
    remaining-=len(b);h.update(b)
  assert h.hexdigest()==sm['streams']['heal'][key]
  results[name]=h.hexdigest()
 print('FULL_SNAPSHOT_PREFIX_VERIFIED',json.dumps(results),flush=True)
print('HEALTH_COMPLETE',flush=True)
PY
