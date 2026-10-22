#1 +60+a
#th2-6y3o-dot-long-pair-launch-20261023-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import json,hashlib,subprocess,sys
from pathlib import Path
base=Path('/mnt/local/_outputs/deep-llms_th2')
root=base/'6y3o-dot-long-pair-20261023-a01'
source=base/'6y3o-dot-router-20261022-a01/source'
assert json.loads((root/'preflight-initial.json').read_text())['status']=='ok'
payload=Path('scripts/bdot_long_pair.py').read_bytes()
assert hashlib.sha256(payload).hexdigest()=='b62242de5bc1dd3e0837934cb1d3fcd6a5f46cfc25049030a970f3ac8082b33e'
with (root/'long_pair.py').open('xb') as f:f.write(payload)
subprocess.run([sys.executable,str(root/'long_pair.py'),'launch','--output',str(root),
 '--source',str(source),'--predecessor',str(base/'6y3o-dot-router-confirm-20261023-a01'),
 '--dense-recipe',str(base/'6y3o-dot-pilot-20261020-a01/prepared/D.train.json'),
 '--router-recipe',str(base/'6y3o-dot-router-20261022-a01/prepared/Partial-25-cosine-s0.train.json'),
 '--host','thiennh-p6-6y3o-worker-0','--session','dot_long_pair_20261023_a01'],check=True)
assert subprocess.check_output(['tmux','display-message','-p','-t','dot_long_pair_20261023_a01','#{pane_dead}'],text=True).strip()=='0'
print('LONG_PAIR_SUBMITTED '+json.dumps({'launch':json.loads((root/'launch.json').read_text()),
 'jobs':[j['name'] for j in json.loads((root/'gated-jobs.json').read_text())['jobs']],
 'driver_sha256':hashlib.sha256(payload).hexdigest()}),flush=True)
REMOTE
