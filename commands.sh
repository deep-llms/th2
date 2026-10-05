#1 +60+a
#th2-tjx3-proxy-gate-sweep-stop-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - <<'PY'
import json,os,signal
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect,process,pidfd_open,pidfd_send_signal
from scripts.train_then_burn import argv
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-fa4-screen-seed42-2500-20261005-a02')
control=Path('/mnt/local/_outputs/deep-llms_th2/proxy-gate-sweep-control-20261005-a01')
control.mkdir(exist_ok=False)
status=inspect();print('BEFORE',json.dumps(status),flush=True)
marker=json.loads(Path('/mnt/local/_gpu_guard/DISABLED').read_text())
assert marker['output']==str(root/'supervised')
owner=process(marker['owner_pid']);args=argv(owner['pid'])
assert 'scripts.train_then_burn' in args and args[args.index('--output')+1]==str(root/'supervised')
chains={}
for worker in status['workers']:
    current=worker;chain=[]
    while current!=owner['pid']:
        assert current>1 and len(chain)<10
        item=process(current);chain.append(item);current=item['ppid']
    chains[str(worker)]=chain
assert len(chains)==8
before=dict(supervisor=owner,argv=args,guard=marker,gpu_status=status,worker_chains=chains,
            queue=json.loads((root/'supervised/run/run.json').read_text()))
(control/'inspection.json').write_text(json.dumps(before,indent=2))
print('VERIFIED_SUPERVISOR',json.dumps(owner),'QUEUE',json.dumps(before['queue']),flush=True)
fd=pidfd_open(owner['pid'])
try:
    assert process(owner['pid'])==owner and argv(owner['pid'])==args
    assert json.loads(Path('/mnt/local/_gpu_guard/DISABLED').read_text())==marker
    pidfd_send_signal(fd,signal.SIGTERM)
    (control/'stop-request.json').write_text(json.dumps(dict(status='SIGTERM_sent_to_verified_supervisor',supervisor=owner),indent=2))
    print('STOP_REQUEST_SENT',owner['pid'],flush=True)
finally:os.close(fd)
PY
sleep 45
cat /mnt/local/_outputs/@PROJECT@/proxy-fa4-screen-seed42-2500-20261005-a02/supervised/supervisor.json
tail -n 15 /mnt/local/_outputs/@PROJECT@/proxy-fa4-screen-seed42-2500-20261005-a02/launch.log
