#1 +30+a
#th2-tjx3-proxy-followup-monitor-20261008-a02
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
date -u
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - <<'PY'
import json,hashlib
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-followup-2500-20261008-a01');run=root/'supervised/run'
def artifact(path,name):
 if path.is_file():
  raw=path.read_bytes();print('ARTIFACT',json.dumps(dict(name=name,sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
for name in ('preflight.json','training-recipe.json','jobs.json','inspection.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/supervisor.json','supervised/burn-verified.json','supervised/run/complete.json'):
 artifact(root/name,name)
for path in (root/'launch.log',):
 if path.is_file():print('LAUNCH_LOG',path.read_text()[-7000:],flush=True)
if (run/'run.json').is_file():
 state=json.loads((run/'run.json').read_text());print('QUEUE',json.dumps(state),flush=True)
 for job in state['jobs'][-3:]:
  p=run/(job['name']+'.log')
  if p.is_file():
   with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6500));print('JOB_LOG',job['name'],f.read().decode(errors='replace'),flush=True)
for pattern in ('seed-*/*/train_config.json','seed-*/*/attention_runtime*.json','seed-*/*/fa4*.json','seed-*/*/result.json','seed-*-validate-*.json','seed-*/*/trainer_state.json','seed-*/*/eval_results.json','seed-*/comparison.json','seed-*/*/checkpoint-*/trainer_state.json'):
 for path in sorted(run.glob(pattern)):artifact(path,str(path.relative_to(run)))
p=root/'supervised/burn.log'
if p.is_file():
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4000));print('BURN_LOG',f.read().decode(errors='replace'),flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
print('MONITOR_FINISHED',flush=True)
PY
