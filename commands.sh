#1 +60+a
#th2-78gg-bottleneck-completion-20260930-2046
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-78gg-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python -u - <<'PY'
import json, hashlib, time
from datetime import datetime, timezone
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect, process
from scripts.train_then_burn import approved_launcher, BURN_HASH, burn_progress
root=Path('/mnt/local/_outputs/deep-llms_th2/deep-bottleneck-5000-20260930-a01')
run=root/'production/run'
print('STATUS_AT',datetime.now(timezone.utc).isoformat(),flush=True)
def artifact(path):
 if path.is_file():
  raw=path.read_bytes()
  print('ARTIFACT_JSON',json.dumps(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),content=raw.decode())),flush=True)
def tail(path):
 if path.is_file():
  with path.open('rb') as f:
   f.seek(max(0,path.stat().st_size-20000))
   lines=f.read().decode(errors='replace').replace('\r','\n').splitlines()
  print('LOG_TAIL',str(path.relative_to(root)),flush=True)
  print('\n'.join([x for x in lines if x.strip()][-18:]),flush=True)
for name in ['production/supervisor.json','production/run/run.json','production/run/complete.json','production/run/comparison.json','production/burn-verified.json']:
 artifact(root/name)
for arm in ['Task-Aware-Align','Consumer-Aware-Align']:
 folder=run/arm
 for name in ['result.json','train_config.json','eval_results.json']:
  artifact(folder/name)
 states=sorted(folder.glob('checkpoint-*/trainer_state.json'),key=lambda p:int(p.parent.name.split('-')[-1]))
 if states:
  path=states[-1]
  state=json.loads(path.read_text())
  print('CHECKPOINT',arm,state['global_step'],json.dumps(state['log_history'][-3:]),flush=True)
  if state['global_step']==5000:
   artifact(path)
   required=['model.safetensors','optimizer.pt','scheduler.pt','training_args.bin','trainer_state.json']+[f'rng_state_{i}.pth' for i in range(8)]
   sizes={name:(path.parent/name).stat().st_size for name in required}
   assert all(sizes.values()),sizes
   print('CHECKPOINT_FILES_VERIFIED',arm,json.dumps(sizes),flush=True)
if (run/'complete.json').exists():
 report=json.loads((run/'complete.json').read_text())
 assert report==json.loads((run/'run.json').read_text())
 assert report['status']=='ok'
 for job in report['jobs']:
  assert job['status']=='ok' and job['returncode']==0
  for item in job['artifacts']:
   raw=(run/item['path']).read_bytes()
   assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
 print('QUEUE_COMPLETION_AND_ARTIFACT_HASHES_VERIFIED',flush=True)
tail(root/'launch.log')
for path in sorted(run.glob('*.log')):tail(path)
tail(root/'production/burn.log')
live=inspect()
print('LIVE_GPU_INSPECTION',json.dumps(live),flush=True)
if (root/'production/burn-verified.json').exists():
 approved={str(Path('resources/llm_pretrain_burn.py').resolve()):BURN_HASH}
 assert len(live['workers'])==8 and all(len(g['pids'])==1 for g in live['gpus'])
 assert not live['guard_disabled']
 assert all(approved_launcher(process(pid)['ppid'],approved) for pid in live['workers'])
 log=root/'production/burn.log'
 first=log.read_text()
 assert burn_progress(first)
 time.sleep(15)
 second=log.read_text()
 assert len(second)>len(first) and burn_progress(second)
 tail(log)
 print('LIVE_EIGHT_GPU_BURN_IDENTITIES_AND_NEW_PROGRESS_VERIFIED',flush=True)
print('INSPECTION_FINISHED',datetime.now(timezone.utc).isoformat(),flush=True)
PY
