#1 +30+a
#th2-wc69-dot-rl-liveness-20261114-a01
set -euo pipefail
printf 'DOT_RL_LIVENESS_20261114_A01_BEGIN\n'
date -u
hostname
uptime
timeout 15 nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used --format=csv,noheader || true
timeout 15 nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv,noheader || true
/mnt/local/conda-py311/envs/dot_pool/bin/python3.11 - <<'PYPROBE'
import json,time,socket
from pathlib import Path
base=Path('/mnt/local/_outputs/deep-llms_th2/dot_rl_20261114')
run=base/'results/dot_rl_20261114/run-a01'
result={'utc':time.time(),'host':socket.gethostname(),'root_exists':base.exists(),'runs':{}}
for p in sorted(run.glob('*/progress.json')):
    entry={'progress':json.loads(p.read_text())}
    for name in ['train_complete.json','complete.json']:
        f=p.parent/name
        entry[name]=json.loads(f.read_text()) if f.exists() else None
    result['runs'][p.parent.name]=entry
for name,p in [('report',run/'report_complete.json'),('supervisor',base/'execution-a01/supervised/supervisor.json'),('native',base/'execution-a01/supervised/burn-verified.json')]:
    result[name]=json.loads(p.read_text()) if p.exists() else None
print('DOT_RL_LIVENESS',json.dumps(result))
PYPROBE
printf 'DOT_RL_LIVENESS_20261114_A01_END\n'
