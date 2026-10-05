#1 +60+a
#th2-tjx3-proxy-fa4-restart-stop-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - <<'PY'
import json,os,signal
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect,process,pidfd_open,pidfd_send_signal
from scripts.train_then_burn import argv
control=Path('/mnt/local/_outputs/deep-llms_th2/proxy-fa4-restart-control-20261005-a01')
receipt=control/'stop-request.json'
assert not receipt.exists()
old=json.loads((control/'inspection.json').read_text())
owner=old['supervisor'];pid=owner['pid']
fd=pidfd_open(pid)
try:
    assert process(pid)==owner
    assert argv(pid)==old['supervisor_argv']
    assert json.loads(Path('/mnt/local/_gpu_guard/DISABLED').read_text())==old['guard']
    status=inspect();assert status['host']==old['host']
    for worker in status['workers']:
        current=worker;seen=set()
        while current!=pid:
            assert current>1 and current not in seen and len(seen)<10
            seen.add(current);current=process(current)['ppid']
    # The supervisor stops only its own child session, then verifies free GPUs
    # and restores the authorized idle burn before releasing its guard.
    assert process(pid)==owner
    pidfd_send_signal(fd,signal.SIGTERM)
    value=dict(status='SIGTERM_sent_to_verified_supervisor',supervisor=owner,gpu_status=status)
    receipt.write_text(json.dumps(value,indent=2))
    print('STOP_REQUEST',json.dumps(value),flush=True)
finally:
    os.close(fd)
PY
sleep 45
cat /mnt/local/_outputs/@PROJECT@/proxy-fa4-screen-seed42-2500-20261005-a01/supervised/supervisor.json
tail -n 15 /mnt/local/_outputs/@PROJECT@/proxy-fa4-screen-seed42-2500-20261005-a01/launch.log
