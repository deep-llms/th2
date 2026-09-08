#1 +60+a
#th2-swt-identify-training-before-authorized-stop-20260908-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-oish-worker-0
test "$PWD" = /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/swt/bin/python - <<'PY'
import json
from pathlib import Path
from scripts.gpu_status import snapshot
from scripts.reclaim_verified_burn import identity
status = snapshot()
print('GPU_SNAPSHOT', json.dumps(status), flush=True)
seen = set()
for gpu in status:
    for pid in gpu['pids']:
        while pid > 1 and pid not in seen:
            seen.add(pid)
            process = identity(pid)
            print('PROCESS_IDENTITY', json.dumps(process), flush=True)
            pid = process['parent']
root = Path('/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_10k_s42_20260908_a02')
for arm in ('B0','A128','A256','A512','C','D'):
    steps = sorted(int(path.name.split('-')[-1]) for path in (root/arm).glob('checkpoint-*') if path.is_dir())
    print('CHECKPOINTS', arm, steps, flush=True)
print('SUCCESS_MARKERS', [(name,(root/name).exists()) for name in ('training_complete.json','burn_verified.json')])
PY
echo SWT_TRAINING_IDENTIFICATION_COMPLETE
