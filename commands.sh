#1 +60+a
#th2-78gg-BFG-10000-completion-20260930-0522
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-78gg-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python - <<'PY'
import hashlib,json,subprocess,time
from scripts.train_then_burn import approved_launcher, BURN_HASH, process, burn_progress
from datetime import datetime,timezone
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/deep-kv-BFG-10000-20260929-a02')
print('STATUS_AT',datetime.now(timezone.utc).isoformat(),flush=True)
paths=[root/p for p in ['production/run/run.json','production/run/complete.json','production/supervisor.json',
 'production/burn-verified.json','production/run/continuation/comparison.json']]
for arm in 'BFG':
 folder=root/'production/run/continuation'/arm
 paths.extend(folder/p for p in ['result.json','eval_results.json','train_config.json','trainer_state.json'])
 config=json.loads((folder/'train_config.json').read_text())
 print('CONFIG',arm,json.dumps({k:config[k] for k in ['pilot','world_size','tokens_per_update','train_fingerprint','eval_fingerprint']}),flush=True)
 states=sorted(folder.glob('checkpoint-*/trainer_state.json'),key=lambda p:int(p.parent.name.split('-')[-1]))
 if states:
  state=json.loads(states[-1].read_text())
  print('CHECKPOINT',arm,states[-1].parent.name,json.dumps({k:state[k] for k in ['global_step','max_steps','epoch']}),flush=True)
  print('RECENT_CHECKPOINT_METRICS',arm,json.dumps(state['log_history'][-3:]),flush=True)
for path in paths:
 if path.is_file():
  raw=path.read_bytes()
  print('ARTIFACT_JSON',json.dumps(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),content=raw.decode())),flush=True)
def tail(path,count=50):
 if path.exists():
  with path.open('rb') as handle:
   handle.seek(max(0,path.stat().st_size-25000))
   lines=handle.read().decode(errors='replace').replace('\r','\n').splitlines()
  print('LOG_TAIL',str(path.relative_to(root)),flush=True)
  print('\n'.join([line for line in lines if line.strip()][-count:]),flush=True)
for name in ['launch.log','production/run/arm-B.log','production/run/arm-F.log','production/run/arm-G.log']:
 tail(root/name)
inspection=inspect()
print('LIVE_GPU_INSPECTION',json.dumps(inspection),flush=True)
if (root/'production/run/complete.json').exists():
 complete=json.loads((root/'production/run/complete.json').read_text())
 assert complete==json.loads((root/'production/run/run.json').read_text())
 assert all(j['status']=='ok' and j['returncode']==0 for j in complete['jobs'])
 for job in complete['jobs']:
  for artifact in job['artifacts']:
   raw=(root/'production/run'/artifact['path']).read_bytes()
   assert hashlib.sha256(raw).hexdigest()==artifact['sha256']
 for arm in 'BFG':
  folder=root/'production/run/continuation'/arm
  result=json.loads((folder/'result.json').read_text())
  state=json.loads((folder/'checkpoint-10000/trainer_state.json').read_text())
  assert result['global_step']==state['global_step']==10000
  assert result['schedule_steps']==state['max_steps']==28600
  for name in ['model.safetensors','optimizer.pt','scheduler.pt','training_args.bin',*[f'rng_state_{rank}.pth' for rank in range(8)]]:
   assert (folder/'checkpoint-10000'/name).stat().st_size>0
 print('COMPLETION_ARTIFACTS_VERIFIED',flush=True)
if (root/'production/burn-verified.json').exists():
 approved={str(Path('resources/llm_pretrain_burn.py').resolve()):BURN_HASH}
 assert len(inspection['workers'])==8 and not inspection['guard_disabled']
 assert all(approved_launcher(process(pid)['ppid'],approved) for pid in inspection['workers'])
 assert burn_progress((root/'production/burn.log').read_text())
 print('LIVE_BURN_IDENTITIES_AND_COLLECTIVES_VERIFIED',flush=True)
print('TMUX_PANE',subprocess.run(['tmux','display-message','-p','-t','deep-kv-BFG-10000-20260929-a02','#{pane_dead}'],capture_output=True,text=True).stdout.strip(),flush=True)
burns=list((root/'production').glob('*burn*.log'))
if burns:
 for path in burns:tail(path,8)
 time.sleep(12)
 for path in burns:tail(path,8)
PY
