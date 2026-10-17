#1 +30+a
#th2-q359-ce7454-five-fits-verified-20261017-a02
set -euo pipefail
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python -u - <<'REMOTE'
import json
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-ce7454-followup-20261017-a02')
study=root/'supervised/run/study'
print('TIME',datetime.now(timezone.utc).isoformat())
for path in [root/'supervised/gpus-free-before-training.json',root/'supervised/supervisor.json',study/'smoke_passed.json',root/'supervised/run/ce7454-study.log']:
    print('RECEIPT',str(path),path.read_text()[-18000:] if path.exists() else 'MISSING')
for name in ['grip_nodown_ibna_s1','grip_nodown_ibnb_s1','grip_nodown_s1','srresnet_s1','srresnet_stats_s1']:
    path=study/(name+'.log')
    print('FIT',name,path.read_text()[-5000:] if path.exists() else 'MISSING')
    p=study/(name+'.command.json')
    print('COMMAND',name,p.read_text() if p.exists() else 'MISSING')
status=snapshot(list(range(8))); print('GPUS',json.dumps(status))
for pid in sorted({p for g in status for p in g['pids']}):
    print('WORKER',pid,(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode())
REMOTE
