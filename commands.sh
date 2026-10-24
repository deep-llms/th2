#1 +30+a
#th2-6y3o-dot-stop-optimized-health-20261025-a04
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import json,hashlib
from pathlib import Path
from scripts.gpu_process import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-stop-optimized-20261025-a02')
read=lambda p:json.loads(p.read_text())
def tail(p,n=2000):
 if not p.is_file():return ''
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
r={'live':inspect(),'supervisor_tail':tail(root/'supervisor.log',10000)}
for rel in ['launch.json','adaptation.json','branches.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/supervisor.json','supervised/burn-verified.json','supervised/run/run.json','optimization.json','supervised/run/numerics.json','supervised/run/optimization-gate.json','supervised/run/capacity.json','supervised/run/ddp/rank0.json','supervised/run/ddp/rank1.json','supervised/run/resume/summary.json','supervised/run/eval_smoke/summary.json','supervised/run/eval_5000/summary.json','supervised/run/eval_10000/summary.json']:
 if (root/rel).is_file():r[rel]=read(root/rel)
if (root/'source/SOURCE_MANIFEST.json').is_file():
 source=root/'source';r['source_matches']=all(hashlib.sha256((source/k).read_bytes()).hexdigest()==v for k,v in read(source/'SOURCE_MANIFEST.json')['files'].items())
 assert r['source_matches']
if 'launch.json' in r:
 r['accelerate_cache_matches']=Path(r['launch.json']['accelerate_cache']).read_bytes()==(root/'source/resources/accelerate_b200.yaml').read_bytes()
 r['accelerate_env']=(root/'accelerate_env.txt').read_text()
r['logs']={str(p.relative_to(root)):tail(p,3500) for p in (root/'supervised/run').glob('*.log')}
r['resume_logs']={p.name:tail(p,3000) for p in (root/'supervised/run/resume').glob('*.log')}
r['training']={}
for p in (root/'prepared/runs').glob('*'):
 entry={'metrics_tail':tail(p/'metrics.jsonl',2500),'checkpoints':sorted(x.name for x in p.glob('checkpoint-*'))}
 for name in ['train_results.json','final_eval.json']:
  if (p/name).is_file():entry[name]=read(p/name)
 if (p/'data_recipe.json').is_file():
  data=read(p/'data_recipe.json');entry['data_slice']=data.get('training_slice');entry['parent_fingerprint']=data['shuffled_fingerprint']
 r['training'][p.name]=entry

r['evaluation_details']={}
for p in (root/'supervised/run').glob('eval_*/*'):
 if p.is_dir():
  entry={}
  for name in ['head.json','calibration.json','result.json']:
   if (p/name).is_file():entry[name]=read(p/name)
  r['evaluation_details'][str(p.relative_to(root))]=entry
 elif p.suffix=='.log':r['logs'][str(p.relative_to(root))]=tail(p,8000)
r['branch_verified']={p.name:read(p) for p in (root/'supervised/run/checks').glob('*.json')}
for p in (root/'prepared/runs').glob('*'):
 cp=p/'checkpoint-10000'
 if cp.is_dir():
  r['training'][p.name]['final_files']={x.name:x.stat().st_size for x in cp.iterdir() if x.is_file()}
  if (cp/'trainer_state.json').is_file():
   ts=read(cp/'trainer_state.json');r['training'][p.name]['trainer_step']=ts.get('global_step')

if 'supervised/burn-verified.json' in r:
 from scripts.native_baseline_pipeline import validate_burn_handoff
 from scripts.train_then_burn import burn_progress
 validate_burn_handoff(r['supervised/supervisor.json'],r['supervised/burn-verified.json'],r['live'])
 r['live_burn_handoff_verified']=True
 r['collective_progress_verified']=burn_progress((root/'supervised/burn.log').read_text())
 r['burn_tail']=tail(root/'supervised/burn.log')
r['evaluation_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'supervised/run').glob('eval_*/summary.json')}
r['initializations']={p.parent.name:read(p) for p in (root/'prepared').glob('*-init/initialization.json')}
r['prepared_configs']={p.name:read(p) for p in (root/'prepared').glob('*.train.json')}
r['benchmarks']={p.name:read(p) for p in (root/'supervised/run').glob('benchmark-*.json')}
r['guard_owner']=Path('/mnt/local/_gpu_guard/DISABLED').read_text() if Path('/mnt/local/_gpu_guard/DISABLED').exists() else None
out=root/'health-a04.json'
with out.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
print('STOP_HEALTH '+json.dumps(r),flush=True)
print('STATUS_FILE_SHA256 '+hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)
REMOTE
