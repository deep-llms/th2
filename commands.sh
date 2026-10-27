#1 +30+a
#th2-6y3o-a2-inputs-readiness-20261028-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import json,socket,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from scripts.a2_prepare_inputs import verify_inputs
from efficient_llm.diagnostic_data import sha256
from scripts.gpu_process import inspect
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
print('READINESS_AT',datetime.now(timezone.utc).isoformat(),flush=True)
plan=json.loads(Path('resources/a2_inputs_20261028.json').read_text())
files=[dict(path=i['path'],exists=Path(i['path']).is_file(),bytes=Path(i['path']).stat().st_size if Path(i['path']).is_file() else None) for i in plan['downloads']]
print('ASSET_FILES',json.dumps(files),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
try:
 verify_inputs(plan,include_models=False)
 for key,item in json.loads(Path('resources/a2_secondary_20261028.json').read_text()).items():
  p=Path(item['path'])
  if p.stat().st_size!=item['bytes'] or sha256(p)!=item['sha256']:raise ValueError('Secondary incomplete: '+key)
except (FileNotFoundError,ValueError) as error:
 print('DATA_INPUTS_NOT_READY',str(error),flush=True)
else:
 print('DATA_INPUT_BYTES_VERIFIED',flush=True)
 # Explicit reviewed source commit; the node's code copy contains no .git.
 subprocess.run([sys.executable,'-u','-m','scripts.launch_a2_inputs',
  '--output','/mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01',
  '--host','thiennh-p6-6y3o-worker-0','--session','a2_inputs_20261028_a01',
  '--code-commit','25899871d2181df7f7ba154c800c8ddf3007b08f'],check=True)
print('READINESS_COMPLETE',flush=True)
PY
