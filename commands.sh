#0
#th2-q359-proxy12-eval-completion-20261014-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 -u - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,time
from scripts.train_then_burn import burn_progress, approved_launcher, BURN_HASH
from scripts.verified_gpu_reclaim import inspect,process
base=Path('/mnt/local/_outputs/deep-llms_th2')
print('AT',datetime.now(timezone.utc).isoformat())
for name in ('q359-proxy12-eval-20261014-a01','q359-proxy12-eval-remaining27-20261014-a01'):
 root=base/name;print('ROOT',name)
 for leaf in ('supervised/supervisor.json','supervised/run/run.json'):
  p=root/leaf
  if p.exists():print('STATE',leaf,p.read_text())
 print('COMPLETE_MARKER',(root/'supervised/run/complete.json').exists())
 for stage in ('numerics','smoke','full'):
  p=root/'supervised/run'/stage/'summary.json'
  if p.exists():
   d=json.loads(p.read_text());print('SUMMARY',stage,d['status'],d.get('diagnostic'),len(d['checkpoints']),hashlib.sha256(p.read_bytes()).hexdigest())
   if stage=='full':
    for ck,row in d['checkpoints'].items():
     print('RESULT',json.dumps(dict(label=d['labels'][ck],checkpoint=ck,summary=row['summary'],checkpoint_sha256=row['checkpoint_sha256'])))
  else:
   plan=p.with_name('plan.json')
   if plan.exists():
    counts=Counter()
    for j in json.loads(plan.read_text()):
     dest=Path(j['output']);meta=dest/'eval_metadata.json'
     counts[json.loads(meta.read_text()).get('status','unknown') if meta.exists() else 'started' if dest.with_suffix('.log').exists() else 'pending']+=1
    print('PROGRESS',stage,dict(counts))
 for leaf in ('supervisor.log','supervised/run/numerics.log','supervised/run/smoke.log','supervised/run/full.log'):
  p=root/leaf
  if p.exists():
   with p.open('rb') as f:f.seek(max(0,p.stat().st_size-9000));s=f.read().decode(errors='replace')
   print('TAIL',leaf,'\n'+'\n'.join(line[:600] for line in s.splitlines()[-5:]))
root=base/'q359-proxy12-eval-remaining27-20261014-a01'
status=inspect();print('LIVE_GPU',json.dumps(status))
p=root/'supervised/burn-verified.json'
if p.exists():
 burn=json.loads(p.read_text());print('BURN_RECEIPT',json.dumps(burn))
 log=root/'supervised/burn.log'
 print('BURN_PROGRESS_VALID',burn_progress(log.read_text()),'LOG_AGE_SECONDS',time.time()-log.stat().st_mtime)
 expected={p for gpu in burn['gpus'] for p in gpu['pids']}
 print('BURN_WORKERS_MATCH',set(status['workers'])==expected)
 print('BURN_PARENTS_APPROVED',all(approved_launcher(process(p)['ppid'],{str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}) for p in status['workers']))
PY
