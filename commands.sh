#1 +60+a
#th2-q359-ce7454-correct-parent-stop-20261017-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python -u - <<'REMOTE'
import json, os, signal, time
from pathlib import Path
from scripts.gpu_status import snapshot
from scripts.verified_gpu_reclaim import process, pidfd_open, pidfd_send_signal
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-ce7454-followup-20261017-a01')
guard=Path('/mnt/local/_gpu_guard/DISABLED')
receipt=root/'supervised/supervisor.json'
print('GPU_BEFORE',json.dumps(snapshot(list(range(8)))),flush=True)
for _ in range(60):
    if receipt.exists(): break
    if (root/'launch.json').exists(): print('LAUNCH', (root/'launch.json').read_text(),flush=True)
    time.sleep(3)
assert receipt.exists(), 'No supervisor receipt: no signals sent'
r=json.loads(receipt.read_text())
if 'finished_at' not in r:
    raw=guard.read_text(); g=json.loads(raw)
    assert g['output']==str(root/'supervised'), 'Another owner: refuse'
    pid=g['owner_pid']; identity=process(pid)
    argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
    assert b'scripts.train_then_burn' in argv and str(root/'supervised').encode() in argv, argv
    fd=pidfd_open(pid)
    try:
        assert process(pid)==identity and guard.read_text()==raw
        print('STOP_VERIFIED_SUPERVISOR',json.dumps(identity),flush=True)
        pidfd_send_signal(fd,signal.SIGTERM)
    finally: os.close(fd)
for _ in range(120):
    r=json.loads(receipt.read_text())
    if 'finished_at' in r: break
    time.sleep(3)
print('SUPERVISOR',json.dumps(r),flush=True)
print('GUARD',guard.read_text() if guard.exists() else 'absent',flush=True)
for path in [root/'supervised/burn-verified.json',root/'supervised/burn.log']:
    if path.exists(): print(str(path),path.read_text()[-16000:],flush=True)
print('GPU_AFTER',json.dumps(snapshot(list(range(8)))),flush=True)
assert 'finished_at' in r and 'burn' in r and 'handoff_error' not in r
assert not guard.exists()
print('CORRECT_PARENT_STOP_AND_BURN_VERIFIED',flush=True)
REMOTE
