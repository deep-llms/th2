#1 +30+a
#th2-6y3o-dot-long-pair-final-20261023-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import json,hashlib
from pathlib import Path
from scripts.gpu_process import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-long-pair-20261023-a01')
read=lambda p:json.loads(Path(p).read_text())
def tail(p,n=5000):
 if not p.is_file():return ''
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
meta=read(root/'long-pair.json');source=Path(meta['source'])
r={'live':inspect(),'supervisor_tail':tail(root/'supervisor.log',3000)}
for rel in ['launch.json','long-pair.json','supervised/supervisor.json','supervised/burn-verified.json','supervised/run/complete.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/run.json','supervised/run/long-pair-check.json','supervised/run/long-pair-report.json']:
 if (root/rel).is_file():r[rel]=read(root/rel)
r['source_matches']=all(hashlib.sha256((source/k).read_bytes()).hexdigest()==v for k,v in read(source/'SOURCE_MANIFEST.json')['files'].items())
assert r['source_matches']
r['accelerate_cache_matches']=Path(r['launch.json']['accelerate_cache']).read_bytes()==(source/'resources/accelerate_b200.yaml').read_bytes()
assert r['accelerate_cache_matches']
r['accelerate_env']=(root/'accelerate_env.txt').read_text()
r['recipes']={};r['runs']={}
for arm in ['D','Partial-25-cosine']:
 name=arm+'-s0';t=read(root/'prepared'/(name+'.train.json'));old=read(meta['parent_recipes'][arm])
 changes={k:[old.get(k),t.get(k)] for k in t.keys()|old.keys() if t.get(k)!=old.get(k)}
 assert set(changes)<={'config_name','output_dir','keep_checkpoint_steps','max_steps','warmup_steps'}
 assert read(t['config_name'])==read(old['config_name'])
 assert t['max_steps']==10000 and t['warmup_steps']==200 and t['seed']==t['data_seed']==0
 assert t.get('keep_checkpoint_steps') is None and t.get('save_total_limit') is None
 assert t.get('resume_from_checkpoint') is None and t.get('model_name_or_path') is None
 r['recipes'][name]={'train':t,'model':read(t['config_name']),'parent_differences':changes,'parent_model_identical':True}
 run=Path(t['output_dir']);v={};r['runs'][name]=v
 for rel in ['train_results.json','final_eval.json','train_config.json']:
  if (run/rel).is_file():v[rel]=read(run/rel)
 if (run/'data_recipe.json').is_file():
  v['data_matches_parent']=read(run/'data_recipe.json')==read(Path(old['output_dir'])/'data_recipe.json')
  assert v['data_matches_parent']
 rows=[]
 for line in tail(run/'metrics.jsonl',20000).splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
 v['latest_metrics']=rows[-3:]
 v['attempt_log_tail']=tail(root/'supervised/run'/(name+'.verified.attempt1.log'),2500)
 v['checkpoints']=[p.name for p in run.glob('checkpoint-*')]
 if v['checkpoints']:
  cp=max(run.glob('checkpoint-*'),key=lambda p:int(p.name[11:]))
  v['checkpoint_files']={p.name:p.stat().st_size for p in cp.iterdir() if p.is_file()}
  if (cp/'trainer_state.json').exists():v['checkpoint_step']=read(cp/'trainer_state.json')['global_step']
for name, v in r['runs'].items():
 for suffix in [name+'.verified.json', name+'/result.json']:
  if (root/'supervised/run'/suffix).is_file():v[suffix]=read(root/'supervised/run'/suffix)
 run=Path(r['recipes'][name]['train']['output_dir'])
 v['retained_checkpoints']={}
 for cp in sorted(run.glob('checkpoint-*'),key=lambda p:int(p.name[11:])):
  v['retained_checkpoints'][cp.name]={p.name:p.stat().st_size for p in cp.iterdir() if p.is_file() and (p.name in ['optimizer.pt','scheduler.pt','trainer_state.json'] or p.name.startswith('rng_state'))}
 if (run/'checkpoint-10000/trainer_state.json').is_file():
  v['periodic_evals']=[x for x in read(run/'checkpoint-10000/trainer_state.json')['log_history'] if 'eval_loss' in x]
if (root/'supervised/burn-verified.json').is_file():
 from scripts.native_baseline_pipeline import validate_burn_handoff
 from scripts.train_then_burn import burn_progress
 try:
  validate_burn_handoff(r['supervised/supervisor.json'], r['supervised/burn-verified.json'], r['live'])
  r['live_burn_handoff_verified']=True
 except Exception as e:r['live_burn_handoff_error']=str(e)
 burn_text=(root/'supervised/burn.log').read_text()
 r['burn_collective_progress_verified']=burn_progress(burn_text)
 r['burn_tail']=burn_text[-4000:]
r['queue_jobs']=[j['name'] for j in read(root/'gated-jobs.json')['jobs']]
out=root/'final-status-20261023-a01.json'
with out.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
print('LONG_PAIR_FINAL '+json.dumps(r),flush=True)
print('STATUS_FILE_SHA256 '+hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)
REMOTE
