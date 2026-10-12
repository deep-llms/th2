#1 +30+a
#th2-q359-expanded-eval-preflight-20261012-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
date -u
/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 -u - <<'PYREMOTE'
import importlib.metadata as md,json,shutil,socket,subprocess
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
print('HOST',socket.gethostname(),flush=True)
print('INSPECTION',json.dumps(inspect()),flush=True)
print('VERSIONS',json.dumps({p:md.version(p) for p in ('torch','transformers','accelerate','datasets','lm_eval','flash-attn-4','pyarrow')}),flush=True)
print('DISK',shutil.disk_usage('/mnt/local'),flush=True)
base=Path('/mnt/local/_outputs/deep-llms_th2')
for root in sorted(base.glob('q359-*')):
 if any(s in root.name for s in ('resume20k','10k','downstream','cooldown')):
  print('OUTPUT',str(root),flush=True)
  for state in (root/'supervised/supervisor.json',root/'supervised/run/run.json'):
   if state.is_file():
    r=json.loads(state.read_text());print('STATE',str(state),json.dumps({k:r.get(k) for k in ('status','training_status','training_returncode','burn')}),flush=True)
  for checkpoint in sorted(root.glob('supervised/run/seed-1042/*/checkpoint-*')):
   if checkpoint.name in ('checkpoint-10000','checkpoint-20000'):
    print('CHECKPOINT',str(checkpoint),json.loads((checkpoint/'trainer_state.json').read_text())['global_step'],(checkpoint/'model.safetensors').stat().st_size,flush=True)
for base in ('/mnt/local/_data/deep-llms_th2','/mnt/local/_models/deep-llms_th2'):
 print('INPUT_DIRS',base,[str(p) for p in Path(base).glob('*')],flush=True)
print('TMUX',subprocess.run(['tmux','list-sessions'],capture_output=True,text=True).stdout,flush=True)
print('PREFLIGHT_READ_ONLY_COMPLETE',flush=True)
PYREMOTE
