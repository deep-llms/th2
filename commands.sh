#1 +60+a
#th2-tjx3-proxy-gate-sweep-burn-verify-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - <<'PY'
import json,hashlib
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import burn_progress,approved_launcher,BURN_HASH
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-gate-sweep-20261005-a01')
state=inspect();print('GPU_STATUS',json.dumps(state),flush=True)
for name in ('supervised/supervisor.json','supervised/burn-verified.json'):
    raw=(root/name).read_bytes();print('ARTIFACT',json.dumps(dict(path=name,sha256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))),flush=True)
terminal=json.loads((root/'supervised/supervisor.json').read_text())
assert terminal['training_status']=='ok' and terminal['burn']['collective_progress_verified']
assert not state['guard_disabled'] and len(state['workers'])==8
assert all(approved_launcher(process(pid)['ppid'],{str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}) for pid in state['workers'])
log=(root/'supervised/burn.log').read_text()
assert burn_progress(log)
print('BURN_TAIL',log[-4500:],flush=True)
print('BURN_VERIFIED_LIVE',state['time'],flush=True)
PY
