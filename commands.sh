#1 +30+a
#th2-wc69-dot-pool-health-20261112-a11
set -euo pipefail
cd /mnt/local/deep-llms_th2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
TASK_PYTHON=/mnt/local/conda-py311/envs/dot_pool/bin/python3.11
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
"$TASK_PYTHON" - <<'PYMORE'
import json
from pathlib import Path
from efficient_llm.dot_pool.data import sha
root=Path('/mnt/local/_outputs/deep-llms_th2/dot_pool_20261112/run-a03')
small={}
for p in sorted((root/'tier1').glob('*.json')):
 v=json.loads(p.read_text());f=Path(v['path']);v['actual_size_matches']=f.stat().st_size==v['bytes'];v['actual_hash_matches']=sha(f)==v['sha256'];small[p.name]=v
files={}
for pattern in ['calibration_reuse.json','budget.json','*/train_complete.json','*/final_eval.json','*/final_state/complete.json','*/pre_decay/complete.json']:
 for p in sorted(root.glob(pattern)):files[str(p.relative_to(root))]=dict(sha256=sha(p),value=json.loads(p.read_text()))
print('DOT_RESULTS_JSON',json.dumps(dict(tier1=small,files=files)))
PYMORE
"$TASK_PYTHON" - <<'PYE1'
import json,math
from pathlib import Path
from efficient_llm.dot_pool.data import sha
root=Path('/mnt/local/_outputs/deep-llms_th2/dot_pool_20261112/run-a03');runs={}
for p in sorted(root.glob('E1-*')):
 if not p.is_dir():continue
 row={};metrics=p/'metrics.jsonl'
 if metrics.exists():
  n=0;bad=[]
  for line in metrics.read_text().splitlines():
   x=json.loads(line);n+=1
   if x.get('grad_norm') is None or not math.isfinite(x['nll']):bad.append(x['step'])
  row.update(logged_rows=n,logged_nonfinite_steps=bad)
 evaluations=sorted(p.glob('eval-*.json'),key=lambda f:int(f.stem.split('-')[-1]))
 if evaluations:
  f=evaluations[-1];row['latest_eval']=dict(path=str(f),sha256=sha(f),value=json.loads(f.read_text()))
 markers=[]
 for f in sorted(p.glob('checkpoint-*/complete.json')):
  value=json.loads(f.read_text());members={name:(f.parent/name).is_file() and (f.parent/name).stat().st_size==rec['bytes'] for name,rec in value['files'].items()};markers.append(dict(path=str(f),marker_sha256=sha(f),step=value['step'],actual_member_sizes_match=members))
 row['checkpoints']=markers;runs[p.name]=row
print('DOT_E1_JSON',json.dumps(runs))
PYE1
