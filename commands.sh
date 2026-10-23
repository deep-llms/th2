#1 +30+a
#th2-6y3o-dot-downstream-health-20261023-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/eval/bin/python3.11 - <<'REMOTE'
import json,hashlib
from pathlib import Path
from scripts.gpu_process import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-downstream-20261023-a01')
read=lambda p:json.loads(p.read_text())
def tail(p,n=2000):
 if not p.is_file():return ''
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
r={'live':inspect(),'supervisor_tail':tail(root/'supervisor.log')}
for rel in ['launch.json','preparation.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/supervisor.json','supervised/burn-verified.json','supervised/run/run.json','supervised/run/adapter-check.json','supervised/run/smoke/summary.json','supervised/run/full/summary.json']:
 if (root/rel).is_file():r[rel]=read(root/rel)
if (root/'source/SOURCE_MANIFEST.json').is_file():
 source=root/'source';r['source_matches']=all(hashlib.sha256((source/k).read_bytes()).hexdigest()==v for k,v in read(source/'SOURCE_MANIFEST.json')['files'].items())
 assert r['source_matches']
if 'launch.json' in r:
 r['accelerate_cache_matches']=Path(r['launch.json']['accelerate_cache']).read_bytes()==(root/'source/resources/accelerate_b200.yaml').read_bytes()
 r['accelerate_env']=(root/'accelerate_env.txt').read_text()
r['logs']={str(p.relative_to(root)):tail(p,1500) for p in (root/'supervised/run').glob('*.log')}
r['workers']={}
for stage in ['smoke','full']:
 for arm in ['D','Partial-25-cosine']:
  for group in ['commonsense','science','reading','language']:
   rel=f'supervised/run/{stage}/{arm}/{group}'
   p=root/(rel+'.json');r['workers'][rel]={'result_bytes':p.stat().st_size if p.exists() else None,'log_tail':tail(root/(rel+'.log'),1000)}
if 'supervised/burn-verified.json' in r:
 from scripts.native_baseline_pipeline import validate_burn_handoff
 from scripts.train_then_burn import burn_progress
 validate_burn_handoff(r['supervised/supervisor.json'],r['supervised/burn-verified.json'],r['live'])
 r['live_burn_handoff_verified']=True
 r['collective_progress_verified']=burn_progress((root/'supervised/burn.log').read_text())
 r['burn_tail']=tail(root/'supervised/burn.log')
out=root/'health-a01.json'
with out.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
print('DOWNSTREAM_HEALTH '+json.dumps(r),flush=True)
print('STATUS_FILE_SHA256 '+hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)
REMOTE
