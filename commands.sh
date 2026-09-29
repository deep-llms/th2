#1 +60+a
#th2-78gg-BFG-completion-check-20260929-1150
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-78gg-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python - <<'PY'
import hashlib, json, re, socket, time
from datetime import datetime, timezone
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect, process
from scripts.train_then_burn import approved_launcher, BURN_HASH, burn_progress
root=Path('/mnt/local/_outputs/deep-llms_th2/deep-kv-BFG-5000-20260929-a01')
print('COMPLETION_CHECK',datetime.now(timezone.utc).isoformat(),socket.gethostname(),flush=True)
names=['recipe.json','production/supervisor.json','production/run/run.json','production/run/complete.json',
       'production/run/comparison.json','production/burn-verified.json']
for arm in 'BFG':
 names.extend(f'production/run/{arm}/{name}' for name in ['result.json','train_config.json','trainer_state.json'])
for name in names:
 path=root/name
 if path.exists():
  raw=path.read_bytes()
  print('ARTIFACT',name,'SHA256',hashlib.sha256(raw).hexdigest(),flush=True)
  print(raw.decode(),flush=True)
 else:print('ABSENT',name,flush=True)
for arm in 'BFG':
 path=root/'production/run'/f'arm-{arm}.log'
 if path.exists():
  with path.open('rb') as f:
   f.seek(max(0,path.stat().st_size-10000))
   lines=f.read().decode(errors='replace').replace('\r','\n').splitlines()
  print('LOG_TAIL',arm,flush=True)
  print('\n'.join([line for line in lines if line.strip()][-12:]),flush=True)
status=inspect()
print('LIVE_GPU_INSPECTION',json.dumps(status),flush=True)
if (root/'production/burn-verified.json').exists():
 samples=[]
 for sample in range(2):
  text=(root/'production/burn.log').read_text()
  matches=re.findall(r'gpu_burn_progress rank=0 completed_cycles=(\d+) .*?completed_collective_payload_gib=([\d.]+)',text)
  progress={'at':datetime.now(timezone.utc).isoformat(),'all_ranks_ready_and_progressing':burn_progress(text),
            'cycles':int(matches[-1][0]) if matches else None,'payload_gib':float(matches[-1][1]) if matches else None}
  samples.append(progress)
  if sample==0:time.sleep(12)
 live=inspect()
 approved={str(Path('resources/llm_pretrain_burn.py').resolve()):BURN_HASH}
 workers=live['workers']
 verified=(len(workers)==8 and all(len(g['pids'])==1 for g in live['gpus'])
           and workers==status['workers'] and all(approved_launcher(process(p)['ppid'],approved) for p in workers)
           and all(p['all_ranks_ready_and_progressing'] for p in samples)
           and samples[1]['cycles']>samples[0]['cycles'] and samples[1]['payload_gib']>samples[0]['payload_gib'])
 print('LIVE_BURN_VERIFICATION',json.dumps({'verified':verified,'samples':samples,'inspection':live}),flush=True)
PY
