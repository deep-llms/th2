#1 +60+a
#th2-78gg-BFG-resume-startup-check-20260929-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-78gg-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python - <<'PY'
import hashlib, json, socket, subprocess
from datetime import datetime, timezone
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/deep-kv-BFG-5000-20260929-a01')
print('STARTUP_CHECK', datetime.now(timezone.utc).isoformat(), socket.gethostname(), flush=True)
for name in ['preflight.json','recipe.json','production/reclaim.json','production/gpus-free-before-training.json',
             'production/supervisor.json','production/run/resume_inputs.json','production/run/run.json']:
 path=root/name
 if path.exists():
  raw=path.read_bytes()
  print('ARTIFACT',name,'SHA256',hashlib.sha256(raw).hexdigest(),flush=True)
  print(raw.decode(),flush=True)
 else:
  print('NOT_YET_PRESENT',name,flush=True)
for arm in 'BFG':
 path=root/'production/run'/arm/'train_config.json'
 if path.exists():
  config=json.loads(path.read_text())
  print('CONFIG',arm,json.dumps({key:config[key] for key in ['pilot','world_size','tokens_per_update','train_fingerprint','eval_fingerprint']}),flush=True)
  print('TRAINING_CONFIG',arm,json.dumps({k:config['training'][k] for k in ['max_steps','warmup_steps','learning_rate','per_device_train_batch_size','gradient_accumulation_steps','seed','ignore_data_skip']}),flush=True)
for name in ['launch.log','production/run/stage-resume.log','production/run/arm-B.log','production/run/arm-F.log','production/run/arm-G.log']:
 path=root/name
 if path.exists():
  with path.open('rb') as f:
   f.seek(max(0,path.stat().st_size-20000))
   lines=f.read().decode(errors='replace').replace('\r','\n').splitlines()
  print('LOG_TAIL',name,flush=True)
  print('\n'.join([line for line in lines if line.strip()][-45:]),flush=True)
print('LIVE_GPU_INSPECTION',json.dumps(inspect()),flush=True)
print('TMUX_PANE',subprocess.run(['tmux','display-message','-p','-t','deep-kv-BFG-5000-20260929-a01','#{pane_dead}'],capture_output=True,text=True).stdout.strip(),flush=True)
PY
