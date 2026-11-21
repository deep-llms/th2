#1 +360+a
#th2-ohmv-layer-adaptation-requalify-continue-20261121-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD:$PWD/tests/warm_order"
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
"$TASK_PYTHON" -m unittest discover -s tests/warm_order -p 'test_adaptation*.py' -v
"$TASK_PYTHON" -m scripts.launch_layer_adaptation_requalify --commit f5d5e95dec417345beb2b1ed8973b4a1e69eaa74 --prepared-sha256 67fcf590bb4386fb6490ef7a5e0fada2c9312066e39094bea058cd7885c0192e --source-sha256 efbc0137d6793961da73f09ebe92fe0c41b24dd78ad6c39f9967c9bac201e057
TASK_GATE_SHA=$("$TASK_PYTHON" - <<'PY'
import json,sys,time
from pathlib import Path
from scripts.gpu_process import inspect,process,ownership
from efficient_llm.warm_order.data import sha
base=Path('/mnt/local/_outputs/deep-llms_th2/layer_adaptation_20261121_gpu_gate_a03')
for _ in range(60):
 if (base/'terminal.json').exists():break
 time.sleep(5)
if not (base/'terminal.json').exists():raise RuntimeError('Gate still running; retrieve existing session, no relaunch')
r=json.loads((base/'terminal.json').read_text())
if r.get('success') is not True or r.get('native_verified') is not True:
 print('ADAPTATION_GATE_FAILED',json.dumps(r),file=sys.stderr)
 for name in ('failure.json','qualify-B-s1.log'):
  p=base/name
  if p.exists():print(p.read_text()[-3000:],file=sys.stderr)
 raise RuntimeError('Gate failure; full continuation prohibited')
complete=json.loads((base/'qualification-complete.json').read_text());digest=sha(base/'gpu-admission.json')
assert complete['ready'] is True and digest==complete['gpu_admission_sha256']
prior=json.loads((base/'native-verified.json').read_text())['samples'][-1];now=inspect()
assert not Path('/mnt/local/_gpu_guard/DISABLED').exists()
assert ownership(prior['gpus'])==ownership(now['gpus']) and all(process(int(pid))==v for pid,v in prior['processes'].items())
print('ADAPTATION_GATE_READY',json.dumps(dict(sha256=digest,terminal=r,precontinuation_native_verified=True)),file=sys.stderr,flush=True)
print(digest)
PY
)
[[ "$TASK_GATE_SHA" =~ ^[a-f0-9]{64}$ ]]
"$TASK_PYTHON" -m scripts.launch_layer_adaptation --commit f5d5e95dec417345beb2b1ed8973b4a1e69eaa74 --prepared-sha256 67fcf590bb4386fb6490ef7a5e0fada2c9312066e39094bea058cd7885c0192e --source-sha256 efbc0137d6793961da73f09ebe92fe0c41b24dd78ad6c39f9967c9bac201e057 --gate-sha256 "$TASK_GATE_SHA"
