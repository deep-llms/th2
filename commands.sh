#1 +60+a
#th2-78gg-BFG-resume-preflight-20260929-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/local/conda-py311/envs/train_env/bin/python - <<'CHECK'
import json, shutil, socket
from datetime import datetime, timezone
from pathlib import Path
import torch
from safetensors import safe_open
base=Path('/mnt/local/_outputs/deep-llms_th2')
roots={'B':base/'deep-kv-2500-20260928-a02/production/run/B',
       'F':base/'deep-kv-FG-2500-20260928-a01/production/run/F',
       'G':base/'deep-kv-FG-2500-20260928-a01/production/run/G'}
print('RESUME_PREFLIGHT',datetime.now(timezone.utc).isoformat(),socket.gethostname(),flush=True)
configs=[]
for arm,root in roots.items():
 checkpoint=root/'checkpoint-2500'
 state=json.loads((checkpoint/'trainer_state.json').read_text())
 result=json.loads((root/'result.json').read_text())
 config=json.loads((root/'train_config.json').read_text())
 assert state['global_step']==result['global_step']==2500 and state['max_steps']==28600
 assert config['pilot']['arm']==arm and config['world_size']==8
 assert result['input_tokens']==2621440000
 for name in ['model.safetensors','optimizer.pt','scheduler.pt','training_args.bin',*[f'rng_state_{i}.pth' for i in range(8)]]:
  assert (checkpoint/name).is_file() and (checkpoint/name).stat().st_size>0,name
 scheduler=torch.load(checkpoint/'scheduler.pt',map_location='cpu',weights_only=True)
 assert scheduler['last_epoch']==2500
 optimizer=torch.load(checkpoint/'optimizer.pt',map_location='meta',weights_only=True,mmap=True)
 assert optimizer['state'] and optimizer['param_groups']
 assert all('exp_avg' in v and 'exp_avg_sq' in v and 'step' in v for v in optimizer['state'].values())
 assert [g['lr'] for g in optimizer['param_groups']]==scheduler['_last_lr']
 with safe_open(checkpoint/'model.safetensors',framework='pt',device='cpu') as weights:
  tensors=len(weights.keys())
 assert tensors>0
 print('RESUMABLE',json.dumps(dict(arm=arm,checkpoint=str(checkpoint),step=state['global_step'],
       scheduler_last_epoch=scheduler['last_epoch'],learning_rates=scheduler['_last_lr'],
       optimizer_states=len(optimizer['state']),weight_tensors=tensors,
       checkpoint_bytes=sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file()),
       train_fingerprint=config['train_fingerprint'],eval_fingerprint=config['eval_fingerprint'])),flush=True)
 config['pilot'].pop('arm');configs.append(config)
 del optimizer
assert all(c==configs[0] for c in configs)
print('MATCHED_CONFIGS',True,'FREE_GIB',shutil.disk_usage(base).free/2**30,flush=True)
print('RESUME_PREFLIGHT_PASSED',flush=True)
CHECK
