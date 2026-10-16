#1 +15+a
#th2-q359-teacher-first-inspect-20261016-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 -u - <<'PY'
import json,time
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
from scripts.train_then_burn import digest,burn_progress
print('INSPECTION',json.dumps(inspect()))
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-tuning-eval-20261016-a01/supervised')
print('SUPERVISOR', (root/'supervisor.json').read_text())
print('BURN_PROGRESS',json.dumps(burn_progress((root/'burn.log').read_text())))
print('BURN_AGE',time.time()-(root/'burn.log').stat().st_mtime)
ck=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-12layer-screen22-20261013-a01/supervised/run/seed-42/A/checkpoint-2500')
print('CHECKPOINT_HASH',digest(ck/'model.safetensors'))
print('RECIPE',(ck.parent/'train_config.json').read_text())
PY
