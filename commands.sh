#1 +30+a
#th2-6y3o-native12-baseline-health-20261020-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
python3 - <<'PY'
from pathlib import Path
import datetime,json,subprocess,resource
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-native12-cx-en-20261020-a02')
print('OBSERVED_UTC',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
print('RESOURCE_LIMITS',json.dumps({'open_files':resource.getrlimit(resource.RLIMIT_NOFILE),'processes':resource.getrlimit(resource.RLIMIT_NPROC),'mem_available':next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))}),flush=True)
for name in ['preparation-failed.json','assets-verified.json','cache-ready.json','launch.json','supervised/run/run.json','supervised/supervisor.json','supervised/burn-verified.json','supervised/run/resume-check/complete.json','supervised/run/real-smoke25.json','supervised/run/real-smoke50.json','supervised/run/baseline-5000.json']:
 p=root/name
 if not p.is_file():continue
 j=json.loads(p.read_text())
 if name=='assets-verified.json':j={k:j[k] for k in ['status','files_verified','splits']}
 if name=='cache-ready.json':j={k:j[k] for k in ['status','seconds','train_sequences','validation_sequences']}
 if name=='supervised/run/run.json':j={**{k:v for k,v in j.items() if k!='jobs'},'jobs':[{k:v for k,v in x.items() if k!='artifacts'} for x in j['jobs']]}
 if name.endswith(('-smoke25.json','-smoke50.json','-5000.json')):j={k:v for k,v in j.items() if k!='files'}
 print('RECEIPT',name,json.dumps(j),flush=True)
for p in [Path(str(root)+'.prepare.log'),root/'supervisor.log',*(root/'supervised/run').glob('*.log')]:
 if not p.is_file():continue
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-32768));lines=f.read().decode(errors='replace').splitlines()
 print('LOG_TAIL',str(p),flush=True)
 print('\n'.join(lines[-12:]),flush=True)
for session in ['native12_prepare_20261020_a02','native12_train_20261020_a02']:
 exists=subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0
 if exists:
  r=subprocess.run(['tmux','display-message','-p','-t',session,'#{session_name} pane_dead=#{pane_dead} pane_pid=#{pane_pid}'],capture_output=True,text=True)
  print('SESSION',session,r.returncode,r.stdout.strip(),flush=True)
 else:print('SESSION',session,'absent',flush=True)

base=root/'prepared/baseline'
config_path=base/'train_config.json'
if config_path.is_file():
 config=json.loads(config_path.read_text())
 print('BASELINE_MODEL_ARGUMENTS',json.dumps(config['model']),flush=True)
 keys=['max_steps','stop_after','warmup_steps','seed','data_seed','bf16','tf32','gradient_checkpointing','per_device_train_batch_size','gradient_accumulation_steps','save_steps','save_total_limit','eval_steps','resume_from_checkpoint']
 print('BASELINE_TRAIN_ARGUMENTS',json.dumps({k:config['training'].get(k) for k in keys}),flush=True)
recipe_path=base/'training_recipe.json'
if recipe_path.is_file():
 recipe=json.loads(recipe_path.read_text())
 print('BASELINE_MODEL',json.dumps({'class':recipe['model']['class'],'layers':recipe['model']['config']['num_hidden_layers'],'attention_backend':recipe['model']['attention_backend'],'has_bdot':'bdot' in recipe['model']['config']}),flush=True)
 print('BASELINE_EXECUTION',json.dumps(recipe['execution']),flush=True)
 print('BASELINE_RUNTIME',json.dumps(recipe['runtime']),flush=True)
if (base/'data_recipe.json').is_file():
 data=json.loads((base/'data_recipe.json').read_text())
 keys=['policy','block_size','shuffle_seed','document_end','attention','positions','labels','requested_workers','effective_workers']
 print('BASELINE_DATA',json.dumps({k:data.get(k) for k in keys}),flush=True)
checkpoints=sorted(base.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
print('BASELINE_CHECKPOINTS',[p.name for p in checkpoints],flush=True)
if checkpoints:
 checkpoint=checkpoints[-1]
 names=['model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json','training_recipe.json','data_recipe.json','train_config.json']+[f'rng_state_{rank}.pth' for rank in range(8)]
 print('CHECKPOINT_FILES',json.dumps({name:(checkpoint/name).stat().st_size if (checkpoint/name).is_file() else None for name in names}),flush=True)
 if (checkpoint/'trainer_state.json').is_file():
  state=json.loads((checkpoint/'trainer_state.json').read_text())
  print('CHECKPOINT_STATE',json.dumps({k:state.get(k) for k in ['global_step','max_steps','train_batch_size','best_metric']}),flush=True)
  print('CHECKPOINT_METRICS',json.dumps(state['log_history'][-8:]),flush=True)
 if (checkpoint/'training_recipe.json').is_file():
  print('CHECKPOINT_RECIPE_MATCH',json.loads((checkpoint/'training_recipe.json').read_text())==recipe,flush=True)

from scripts.gpu_process import inspect
live=inspect()
print('GPU_LIVE',json.dumps({'time':live['time'],'gpus':live['gpus'],'guard_disabled':live['guard_disabled']}),flush=True)
PY
