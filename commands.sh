#1 +60+a
#th2-6y3o-dot-router-confirm-launch-20261023-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import json,hashlib,subprocess,sys
from pathlib import Path
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-router-confirm-20261023-a01')
assert json.loads((root/'preflight-initial.json').read_text())['status']=='ok'
payload=Path('scripts/bdot_router_confirm.py').read_bytes()
assert hashlib.sha256(payload).hexdigest()=='5c2b86f73899a8f92a5c78b501b986b6b493284d3110f7ba5919978409e3107f'
with (root/'confirmation.py').open('xb') as f:f.write(payload)
subprocess.run([sys.executable,str(root/'confirmation.py'),'launch','--output',str(root),
 '--predecessor','/mnt/local/_outputs/deep-llms_th2/6y3o-dot-router-20261022-a01',
 '--baseline-recipe','/mnt/local/_outputs/deep-llms_th2/6y3o-dot-v3-20261021-a01/prepared/D-s1.train.json',
 '--host','thiennh-p6-6y3o-worker-0','--session','dot_router_confirm_20261023_a01'],check=True)
assert subprocess.check_output(['tmux','display-message','-p','-t','dot_router_confirm_20261023_a01','#{pane_dead}'],text=True).strip()=='0'
print('ROUTER_CONFIRM_SUBMITTED '+json.dumps({'launch':json.loads((root/'launch.json').read_text()),
 'jobs':[j['name'] for j in json.loads((root/'gated-jobs.json').read_text())['jobs']],
 'driver_sha256':hashlib.sha256(payload).hexdigest()}),flush=True)
REMOTE
