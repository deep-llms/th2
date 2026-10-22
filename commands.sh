#1 +30+a
#th2-6y3o-dot-router-confirm-health-20261023-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import json,hashlib
from pathlib import Path
from scripts.gpu_process import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-router-confirm-20261023-a01')
prior=root.parent/'6y3o-dot-router-20261022-a01';source=prior/'source'
read=lambda p:json.loads(Path(p).read_text())
def tail(p,n=10000):
 if not p.exists():return ''
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
r={'live':inspect(),'supervisor_tail':tail(root/'supervisor.log',3000)}
for rel in ['launch.json','confirmation.json','supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/run.json','supervised/run/confirmation-check.json','supervised/run/confirmation-results.json']:
 if (root/rel).is_file():r[rel]=read(root/rel)
r['source_matches']=all(hashlib.sha256((source/k).read_bytes()).hexdigest()==v for k,v in read(source/'SOURCE_MANIFEST.json')['files'].items())
assert r['source_matches']
if 'launch.json' in r:
 r['accelerate_cache_matches']=Path(r['launch.json']['accelerate_cache']).read_bytes()==(source/'resources/accelerate_b200.yaml').read_bytes()
 assert r['accelerate_cache_matches']
r['accelerate_env']=(root/'accelerate_env.txt').read_text() if (root/'accelerate_env.txt').exists() else None
r['recipes']={};r['runs']={}
for arm in ['Partial-25-cosine','Partial-50-cosine-detach']:
 name=arm+'-s1';t=read(root/'prepared'/(name+'.train.json'));old=read(prior/'prepared'/(arm+'-s0.train.json'))
 assert {k for k in t.keys()|old.keys() if t.get(k)!=old.get(k)}=={'seed','config_name','output_dir'}
 m=read(t['config_name']);om=read(old['config_name']);assert m['bdot'].pop('seed')==1 and om['bdot'].pop('seed')==0 and m==om
 r['recipes'][name]=dict(seed=t['seed'],data_seed=t['data_seed'],routing_seed=1,max_steps=t['max_steps'],warmup=t['warmup_steps'],sequence_length=t['block_size'],microbatch=t['per_device_train_batch_size'],accumulation=t['gradient_accumulation_steps'],attention_backend=t['attention_backend'],parent_matches=True)
 run=Path(t['output_dir']);v={};r['runs'][name]=v
 for rel in ['train_results.json','final_eval.json','train_config.json']:
  if (run/rel).is_file():v[rel]=read(run/rel)
 rows=[]
 for line in tail(run/'metrics.jsonl').splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
 v['latest_metrics']=rows[-3:]
 v['attempt_log_tail']=tail(root/'supervised/run'/(name+'.verified.attempt1.log'),1800)
 v['checkpoints']=[p.name for p in run.glob('checkpoint-*')]
r['queue_jobs']=[j['name'] for j in read(root/'gated-jobs.json')['jobs']]
(root/'health-a01.json').write_text(json.dumps(r,indent=2)+'\n')
print('ROUTER_CONFIRM_HEALTH '+json.dumps(r),flush=True)
REMOTE
