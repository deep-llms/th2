#1 +30+a
#th2-6y3o-partial-router-full-health-20261025-a02
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
export PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import hashlib,json,sys
from pathlib import Path
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-partial-route-full-20261025-a01')
source=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-stop-optimized-20261025-a03/source')
sys.path.insert(0,str(source))
from scripts.gpu_process import inspect
from scripts.native_baseline_pipeline import validate_burn_handoff
from scripts.train_then_burn import burn_progress
read=lambda p:json.loads(p.read_text())
def tail(p,size=3000):
    if not p.is_file():return ''
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-size));return f.read().decode(errors='replace')
r=dict(live=inspect(),root_exists=root.is_dir(),supervisor_tail=tail(root/'supervisor.log',6000))
for rel in ['manifest.json','preflight.json','launch.json','supervised/reclaim.json','supervised/gpus-free-before-training.json',
            'supervised/run/run.json','supervised/run/complete.json','supervised/run/summary.json',
            'supervised/supervisor.json','supervised/burn-verified.json']:
    if (root/rel).is_file():r[rel]=read(root/rel)
r['logs']={p.name:tail(p,3500) for p in (root/'supervised/run').glob('*.log')}
r['finished_results']={p.parent.name:read(p) for p in (root/'supervised/run').glob('*/result.json')}
if (root/'launch.json').is_file():
    r['accelerate_env']=(root/'accelerate_env.txt').read_text()
    r['accelerate_cache_matches']=Path(r['launch.json']['accelerate_cache']).read_bytes()==(source/'resources/accelerate_b200.yaml').read_bytes()
if 'supervised/burn-verified.json' in r:
    validate_burn_handoff(r['supervised/supervisor.json'],r['supervised/burn-verified.json'],r['live'])
    r['live_burn_handoff_verified']=True
    r['collective_progress_verified']=burn_progress((root/'supervised/burn.log').read_text())
    r['burn_tail']=tail(root/'supervised/burn.log')
if (root/'supervised/run/summary.json').is_file():
    r['summary_sha256']=hashlib.sha256((root/'supervised/run/summary.json').read_bytes()).hexdigest()
r['guard_owner']=Path('/mnt/local/_gpu_guard/DISABLED').read_text() if Path('/mnt/local/_gpu_guard/DISABLED').exists() else None
out=root/'health-a02.json'
with out.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
print('FULL_ROUTE_HEALTH '+json.dumps(r),flush=True)
print('STATUS_FILE_SHA256 '+hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)
REMOTE
