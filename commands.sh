#1 +60+a
#th2-swt-stop-old-finetune-preserve-all-results-20260909-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-oish-worker-0
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt_eval
date -u
python - <<'PY'
import json, os, signal, time
from pathlib import Path
from scripts.reclaim_verified_burn import identity
from scripts.gpu_status import snapshot, require_free
root = '/mnt/local/_outputs/deep-llms_th2/swt/full_eval_finetune_42ckpt_20260909_a01'
pipeline = identity(199032)
queue = identity(215026)
assert pipeline['start'] == '168650771'
assert pipeline['argv'] == ['bash','scripts/eval_finetune_capacity_b200.sh',root]
assert queue['start'] == '168932038' and queue['parent'] == pipeline['pid']
assert queue['argv'][:4] == ['python','-u','-m','finetune.run_all']
assert queue['argv'][queue['argv'].index('--output-dir')+1] == root+'/finetune'
def processes():
    result = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit() and int(path.name) > 1:
            try:
                result[int(path.name)] = identity(int(path.name))
            except (FileNotFoundError, ProcessLookupError):
                pass
    return result
def signal_exact(record, sig):
    try:
        current = identity(record['pid'])
        assert all(current[k] == record[k] for k in ('pid','start','argv'))
        os.kill(record['pid'], sig)
        print('SIGNAL_EXACT_OWNED_PID', record['pid'], sig.name, flush=True)
    except (FileNotFoundError, ProcessLookupError):
        print('OWNED_PID_ALREADY_EXITED', record['pid'], flush=True)
def check_worker(r):
    assert r['argv'][:4] == ['/mnt/local/conda-py311/envs/swt_eval/bin/python','-u','-m','finetune.train']
    assert Path(r['argv'][r['argv'].index('--output-dir')+1]).parent == Path(root)/'finetune'
before = processes()
direct = [r for r in before.values() if r['parent'] == queue['pid']]
assert direct
for r in direct:
    check_worker(r)
status = snapshot()
assert [g['index'] for g in status] == list(range(8))
assert {p for g in status for p in g['pids']} <= {r['pid'] for r in direct}
print('VERIFIED_PIPELINE_AND_QUEUE', json.dumps([pipeline, queue]), flush=True)
# Stop dispatch temporarily before collecting workers; never signal runner parents/groups.
frozen = []
try:
    for r in (pipeline, queue):
        signal_exact(r, signal.SIGSTOP)
        frozen.append(r)
    time.sleep(.1)
    records = processes()
    direct = [r for r in records.values() if r['parent'] == queue['pid']]
    for r in direct:
        check_worker(r)
        signal_exact(r, signal.SIGSTOP)
        frozen.append(r)
    records = processes()
    owned = {queue['pid']}
    while True:
        expanded = owned | {p for p,r in records.items() if r['parent'] in owned}
        if expanded == owned:
            break
        owned = expanded
    children = [records[p] for p in owned if p != queue['pid']]
    for r in children:
        check_worker(r)
    assert {p for g in snapshot() for p in g['pids']} <= owned
    print('VERIFIED_OWNED_DESCENDANTS', json.dumps(children), flush=True)
    signal_exact(pipeline, signal.SIGKILL)
    signal_exact(queue, signal.SIGKILL)
    for r in children:
        signal_exact(r, signal.SIGKILL)
finally:
    # If a gate fails before termination, do not leave surviving jobs frozen.
    for r in reversed(frozen):
        signal_exact(r, signal.SIGCONT)
time.sleep(30)
require_free(list(range(8)))
for r in (pipeline, queue, *children):
    try:
        now = identity(r['pid'])
        assert now['start'] != r['start'], f'Owned PID still alive: {r["pid"]}'
    except (FileNotFoundError, ProcessLookupError):
        pass
assert (Path(root)/'eval/complete.json').is_file()
assert not (Path(root)/'complete.json').exists()
print('OLD_FINETUNE_STOPPED_ALL_GPUS_FREE_ALL_OUTPUTS_PRESERVED', flush=True)
PY
