#1 +60+a
#th2-78gg-deep-kv-completion-check-20260928-1248
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-78gg-worker-0
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/train_env/bin/python -u - <<'PY'
from datetime import datetime, timezone
import hashlib, json, re, socket, subprocess, time
from pathlib import Path
from scripts.gpu_status import snapshot
from scripts.train_then_burn import approved_launcher, BURN_HASH, burn_progress, process
root = Path('/mnt/local/_outputs/deep-llms_th2/deep-kv-2500-20260928-a02/production')
run = root / 'run'
def read(path):
    return json.loads(path.read_text()) if path.is_file() else None
record = {'host': socket.gethostname(), 'at': datetime.now(timezone.utc).isoformat(),
          'run': read(run/'run.json'), 'complete': read(run/'complete.json'),
          'comparison': read(run/'comparison.json'), 'supervisor': read(root/'supervisor.json'),
          'burn_receipt': read(root/'burn-verified.json'), 'results': {}, 'states': {}}
configs = []
for arm in 'ABCD':
    result = read(run/arm/'result.json')
    record['results'][arm] = result
    state = read(run/arm/'trainer_state.json')
    if state:
        record['states'][arm] = {k: state[k] for k in ('global_step', 'max_steps')}
    config = read(run/arm/'train_config.json')
    if config:
        config['pilot'].pop('arm')
        configs.append(config)
record['matched_configs'] = len(configs)==4 and all(c==configs[0] for c in configs)
record['artifact_checks'] = []
if record['complete']:
    for job in record['complete']['jobs']:
        for artifact in job.get('artifacts', []):
            p = run / artifact['path']
            record['artifact_checks'].append({'path': artifact['path'], 'valid': p.is_file() and
                hashlib.sha256(p.read_bytes()).hexdigest()==artifact['sha256']})
def progress():
    path = root/'burn.log'
    text = path.read_text() if path.exists() else ''
    matches = re.findall(r'gpu_burn_progress rank=0 completed_cycles=(\d+) .*?completed_collective_payload_gib=([\d.]+)', text)
    return (int(matches[-1][0]), float(matches[-1][1])) if matches else None
before = progress()
time.sleep(12)
after = progress()
record['burn_progress_before'] = before
record['burn_progress_after'] = after
record['burn_advancing_now'] = bool(before and after and after[0]>before[0] and after[1]>before[1])
record['gpus_now'] = snapshot(list(range(8)))
workers = sorted({p for g in record['gpus_now'] for p in g['pids']})
source = str(Path('resources/llm_pretrain_burn.py').resolve())
record['gpu_workers_are_reviewed_burn'] = (len(workers)==8 and
    all(len(g['pids'])==1 for g in record['gpus_now']) and
    all(approved_launcher(process(p)['ppid'], {source: BURN_HASH}) for p in workers))
record['guard_disabled'] = Path('/mnt/local/_gpu_guard/DISABLED').exists()
record['observed_at_end'] = datetime.now(timezone.utc).isoformat()
p = root/'status-20260928-1248.json'
with p.open('x') as handle:
    json.dump(record, handle, indent=2, allow_nan=False)
print('STATUS_JSON_BEGIN')
print(json.dumps(record, indent=2, allow_nan=False))
print('STATUS_JSON_END')
PY
