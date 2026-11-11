#1 +30+a
#th2-wc69-dot-pool-handoff-export-20261112-a04
set -euo pipefail
cd /mnt/local/deep-llms_th2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
TASK_PYTHON=/mnt/local/conda-py311/envs/dot_pool/bin/python3.11
"$TASK_PYTHON" -u -m scripts.dot_pool_health --name a02
if test -d /mnt/local/_outputs/deep-llms_th2/dot_pool_20261112/execution-a03; then
  "$TASK_PYTHON" -u -m scripts.dot_pool_health --name a03
fi
"$TASK_PYTHON" - <<'PY'
import json,time
from pathlib import Path
from efficient_llm.dot_pool.data import sha
base=Path('/mnt/local/_outputs/deep-llms_th2/dot_pool_20261112')
guard=Path('/mnt/local/_gpu_guard/DISABLED')
pending=base/'pending-a03';manifest=json.loads((pending/'source-manifest.json').read_text())
errors=[r for r,h in manifest['files'].items() if sha(pending/'source'/r)!=h]
ready=json.loads((base/'data/ready.json').read_text())
locks={sub:sha(base/'data'/sub/'manifest.json')==ready[key] for sub,key in [('e1','kg_manifest_sha256'),('e2','data_manifest_sha256')]}
log=pending/'handoff.log'
print('DOT_LOCKS_JSON',json.dumps(dict(utc=time.time(),guard=json.loads(guard.read_text()) if guard.exists() else None,data_ready_sha256=sha(base/'data/ready.json'),data_manifest_locks=locks,queued_source_errors=errors,queued_source_manifest_sha256=sha(pending/'source-manifest.json'),handoff_log_tail=log.read_text()[-6000:] if log.exists() else None)))
PY

"$TASK_PYTHON" - <<'PYEXPORT'
"""Collect only completed calibration metrics/proofs, never checkpoint tensors."""
import hashlib,json,tarfile,io
from pathlib import Path
base=Path('/mnt/local/_outputs/deep-llms_th2/dot_pool_20261112')
run=base/'run-a02';dest=base/'exports/calibration-a02-small'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
required=['config.json','metrics.jsonl','final_eval.json','train_complete.json','final_state/complete.json']
files=[]
for i in range(2):
 p=run/f'cal-e1-lr{i}'
 if not (p/'train_complete.json').exists():
  print('CALIBRATION_EXPORT_NOT_READY');raise SystemExit(0)
 if json.loads((p/'train_complete.json').read_text())['step']!=38500:raise ValueError('Incomplete calibration')
 for rel in required:
  f=p/rel
  if not f.is_file() or f.is_symlink():raise ValueError('Missing/linked result '+str(f))
  files.append(f)
 files+=list(p.glob('eval-*.json'))
for name in ['tests.json','kernel_tests.json','status.json','phases.jsonl']:
 if (run/name).is_file():files.append(run/name)
files+=list(run.glob('overfit-*.json'))
for name in ['status.json','reuse-proof.json','stop-proof.json','source-manifest.json']:
 if (base/'pending-a03'/name).is_file():files.append(base/'pending-a03'/name)
if any(p.is_symlink() or p.stat().st_size>8*1024*1024 for p in files):raise ValueError('Unexpected linked/large result')
if sum(p.stat().st_size for p in files)>32*1024*1024:raise ValueError('Result set too large')
if dest.exists():raise ValueError('Export already exists; retrieve rather than overwrite')
dest.mkdir(parents=True)
blobs={str(p.relative_to(base)):p.read_bytes() for p in sorted(files)}
if any(len(b)>8*1024*1024 for b in blobs.values()) or sum(map(len,blobs.values()))>32*1024*1024:raise ValueError('Snapshot too large')
members={r:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for r,b in blobs.items()}
manifest=dict(spec_sha256='a0b41acecbd462db6f9277ebec7173fe242d3493d9fb7f2d77b127d88bc97f05',kind='calibration results only; no weights or inputs',members=members)
(dest/'members.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=dest/'calibration_results.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for rel,blob in blobs.items():
  info=tarfile.TarInfo(rel);info.size=len(blob);info.mode=0o644;tar.addfile(info,io.BytesIO(blob))
 tar.add(dest/'members.json',arcname='members.json',recursive=False)
if archive.stat().st_size>16*1024*1024:raise ValueError('Archive too large; no export permitted')
receipt=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=len(members),members_sha256=sha(dest/'members.json'))
(dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('CALIBRATION_SMALL_READY',json.dumps(receipt),flush=True)

PYEXPORT
