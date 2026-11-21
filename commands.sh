#1 +240+a
#th2-ohmv-layer-adaptation-numerical-diagnose-20261121-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
"$TASK_PYTHON" -m scripts.launch_layer_adaptation_diagnose --commit 07f7dc810cb1dc7afc8c0adbcc7e556859aed070 --prepared-sha256 67fcf590bb4386fb6490ef7a5e0fada2c9312066e39094bea058cd7885c0192e --source-sha256 413117416c1d2ca97d69b6dcefcc9ed4a286a9ffc03a12e60835490ad034c142
"$TASK_PYTHON" - <<'PY'
import json,time
from pathlib import Path
from scripts.gpu_process import inspect,process,ownership
from scripts.train_then_burn import GUARD_HASH
from efficient_llm.warm_order.data import sha
base=Path('/mnt/local/_outputs/deep-llms_th2/layer_adaptation_20261121_gpu_diagnose_a01')
for _ in range(60):
 if (base/'terminal.json').exists():break
 time.sleep(5)
r=dict(state={},source_checks={})
for name in ('terminal.json','failure.json','status.json','diagnosis/complete.json','diagnosis/REPORT.json','execution/submitted.json'):
 p=base/name
 if p.exists():r['state'][name]=json.loads(p.read_text())
source=json.loads((base/'execution/source.json').read_text());r['source_checks']={p:sha(base/'execution/source'/p)==h for p,h in source.items()};assert all(r['source_checks'].values())
for name in ('execution/queue.log','inference-B-s1.log'):
 p=base/name
 if p.exists():r[name]=p.read_text()[-2200:]
if r['state'].get('terminal.json',{}).get('native_verified'):
 prior=json.loads((base/'native-verified.json').read_text())['samples'][-1];now=inspect()
 assert not Path('/mnt/local/_gpu_guard/DISABLED').exists() and sha('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
 assert ownership(prior['gpus'])==ownership(now['gpus']) and all(process(int(pid))==v for pid,v in prior['processes'].items())
 r['native_handoff_current_verified']=True
if 'diagnosis/complete.json' in r['state']:
 assert sha(base/'diagnosis/REPORT.json')==r['state']['diagnosis/complete.json']['report_sha256']
print('ADAPTATION_DIAGNOSIS_STATUS',json.dumps(r),flush=True)
PY
