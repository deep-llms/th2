#1 +60+a
#th2-q359-verify-smoke-completion-20261009-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import json
from pathlib import Path
import re
import time
from scripts.verified_gpu_reclaim import inspect, process
from scripts.train_then_burn import approved_launcher, BURN_HASH, burn_progress
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-fa4-runtime-smoke-20261009-a01/supervised')
smoke=json.loads((root/'run/smoke.json').read_text())
supervisor=json.loads((root/'supervisor.json').read_text())
complete=json.loads((root/'run/complete.json').read_text())
assert smoke['status']=='passed' and smoke['world_size']==8
assert supervisor['training_status']=='ok' and supervisor['training_returncode']==0
assert supervisor['burn']['collective_progress_verified'] and 'handoff_error' not in supervisor
print('SMOKE_RESULT',json.dumps(smoke),flush=True)
print('SUPERVISOR_COMPLETED',supervisor['finished_at'],flush=True)
print('QUEUE_COMPLETION',json.dumps(complete),flush=True)
log=root/'burn.log'
before=log.read_text()
assert burn_progress(before)
def cycles(text):
    return int(re.findall(r'gpu_burn_progress rank=0 completed_cycles=(\d+)',text)[-1])
time.sleep(15)
after=log.read_text()
assert cycles(after)>cycles(before)
record=inspect()
assert record['host']=='thiennh-p6-q359-worker-0' and not record['guard_disabled']
assert len(record['workers'])==8 and all(len(g['pids'])==1 for g in record['gpus'])
approved={str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}
assert all(approved_launcher(process(pid)['ppid'],approved) for pid in record['workers'])
print('LIVE_BURN_VERIFIED',json.dumps(dict(workers=record['workers'],cycles_before=cycles(before),cycles_after=cycles(after),guard_enabled=True)),flush=True)
print('FA4_SMOKE_AND_BURN_HANDOFF_CONFIRMED',flush=True)
PYREMOTE
