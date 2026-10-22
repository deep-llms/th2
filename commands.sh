#1 +30+a
#th2-6y3o-dot-router-followup-health-20261022-a02
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import json,hashlib,subprocess
from pathlib import Path
from scripts.gpu_process import inspect,process
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-router-20261022-a01')
old=root.parent/'6y3o-dot-mlp-20261022-a01'
read=lambda p:json.loads(Path(p).read_text())
status=read(root/'wait-status.json');pid=read(root/'wait-owner.json')['pid']
result={'wait_status':status,'live':inspect(),'wait_log_tail':(root/'wait.log').read_text()[-2000:]}
if status['status']=='waiting':
 result['waiter']=process(pid)
 result['waiter_pane_dead']=subprocess.check_output(['tmux','display-message','-p','-t','dot_router_wait_20261022_a01','#{pane_dead}'],text=True).strip()
 assert result['waiter_pane_dead']=='0' and pid not in result['live']['workers']
result['original_queue']=read(old/'supervised/run/run.json')
result['original_source_matches']=all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v
 for p in (old/'prepared/runs').glob('*/training_recipe.json') for k,v in read(p)['code_sha256'].items())
assert result['original_source_matches']
source=root/'source';manifest=read(source/'SOURCE_MANIFEST.json')
result['snapshot_matches']=all(hashlib.sha256((source/k).read_bytes()).hexdigest()==v for k,v in manifest['files'].items())
assert result['snapshot_matches']
result['recipes']={}
for path in sorted((root/'prepared').glob('*.train.json')):
 t=read(path);name=path.name.removesuffix('.train.json');base='-'.join(name.split('-')[:2]);original=read(old/'prepared'/(base+'-s0.train.json'))
 differences={k:[original.get(k),t.get(k)] for k in original.keys()|t.keys() if original.get(k)!=t.get(k)}
 assert set(differences)<= {'config_name','output_dir'},differences
 m=read(t['config_name']);parent=read(old/'prepared'/(base+'-s0.model.json'))
 flags={k:m['bdot'].pop(k) for k in ['router_detach_input','router_metric']}
 assert m==parent
 result['recipes'][name]={'steps':t['max_steps'],'warmup':t['warmup_steps'],'seed':t['seed'],'data_seed':t['data_seed'],
  'sequence_length':t['block_size'],'microbatch':t['per_device_train_batch_size'],'accumulation':t['gradient_accumulation_steps'],
  'attention':t['attention_backend'],'flags':flags,'parent_recipe_matches':True}
result['queue_jobs']=[j['name'] for j in read(root/'gated-jobs.json')['jobs']]
for p in (old/'prepared/runs').glob('*'):
 if (p/'final_eval.json').is_file():result[p.name]={'final_loss':read(p/'final_eval.json')['eval_loss']}
 elif (p/'metrics.jsonl').is_file():result[p.name]={'last_metrics':[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines() if l][-2:]}
for path in ['launch.json','supervised/supervisor.json','supervised/run/run.json','supervised/run/checks/summary.json']:
 if (root/path).is_file():result[path]=read(root/path)
(root/'status-a02.json').write_text(json.dumps(result,indent=2))
print('ROUTER_FOLLOWUP_HEALTH '+json.dumps(result),flush=True)
REMOTE
