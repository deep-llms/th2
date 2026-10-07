#1 +30+a
#th2-tjx3-proxy-remaining-results-20261007-a02
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
date -u
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PY'
import hashlib
import json
from pathlib import Path
from scripts.gpu_status import snapshot
root = Path('/mnt/local/_outputs/deep-llms_th2/proxy-remaining-2500-20261006-a01')
run = root/'supervised/run'
def read(path):
    return json.loads(path.read_text())
def artifact(path, name):
    if path.is_file():
        raw = path.read_bytes()
        print('ARTIFACT', json.dumps(dict(name=name, sha256=hashlib.sha256(raw).hexdigest(), text=raw.decode())), flush=True)
for name in ('supervised/supervisor.json', 'supervised/burn-verified.json', 'supervised/run/complete.json'):
    artifact(root/name, name)
p = run/'run.json'
if p.is_file():
    state = read(p)
    print('QUEUE', json.dumps(state), flush=True)
    for job in state['jobs'][-3:]:
        log = run/(job['name']+'.log')
        if log.is_file():
            print('JOB_LOG', job['name'], log.read_text()[-6000:], flush=True)
for folder in sorted((run/'training/seed-42').glob('*')):
    if not folder.is_dir():
        continue
    for name in ('result.json', 'train_config.json'):
        artifact(folder/name, 'training/'+folder.name+'/'+name)
    checkpoints = sorted(folder.glob('checkpoint-*/trainer_state.json'), key=lambda p: int(p.parent.name.split('-')[-1]))
    states = [folder/'trainer_state.json'] if (folder/'trainer_state.json').is_file() else checkpoints[-1:]
    for path in states:
        d = read(path)
        print('TRAINER', folder.name, json.dumps(dict(path=str(path), step=d['global_step'], history=d['log_history'][-3:])), flush=True)
for path in sorted((run/'checks').glob('*production-validation.json')):
    artifact(path, 'checks/'+path.name)
for path in sorted((run/'checks').glob('*recipe.json')):
    artifact(path, 'checks/'+path.name)
for name in ('training/seed-42/comparison.json', 'training/comparison-seeds.json'):
    artifact(run/name, name)
baseline = Path('/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-fa4-2500-20261005-a01/supervised/run/baseline/seed-42/A')
for name in ('result.json', 'train_config.json'):
    artifact(baseline/name, 'baseline/A/'+name)
print('GPUS', json.dumps(snapshot(list(range(8)))), flush=True)
print('MONITOR_FINISHED', flush=True)
PY
