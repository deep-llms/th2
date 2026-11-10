#1 +20+a
#th2-wc69-family-f10-health-20261111-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
from pathlib import Path
import json,subprocess
from scripts.a2_family_f10_health import health
from scripts.gpu_process import inspect
from efficient_llm.diagnostic_data import sha256
base=Path('/mnt/local/_outputs/deep-llms_th2')
prep=base/'wc69-a2-family-f10-preparation-20261111-a01'
admission=base/'wc69-a2-family-f10-admission-20261111-a01'
for p in [prep/'prepare_recovery_a03.log',prep/'prepared/prepare.log',prep/'prepared/ready.json',admission/'admission.log']:
 print('PREPARATION_ADMISSION',p.name,p.read_text()[-5000:] if p.exists() else 'ABSENT',flush=True)
for name in ['a2_family_f10_prepare_20261111_a03','a2_family_f10_admission_20261111_a01','a2_family_f10_20261111_a01']:
 r=subprocess.run(['tmux','list-panes','-t',name,'-F','#{pane_pid} #{pane_dead} #{pane_dead_status}'],capture_output=True,text=True)
 print('SESSION',name,r.returncode,r.stdout.strip(),flush=True)
root=base/'wc69-a2-family-f10-20261111-a01'
if (root/'config.json').is_file():health(root)
else:print('GPUQUEUE_NOT_CREATED',json.dumps(inspect()),flush=True)
PY
