#1 +30+a
#th2-q359-ce7454-followup-health-20261017-a02
set -euo pipefail
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python -u - <<'REMOTE'
import json
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_status import snapshot
from scripts.verified_gpu_reclaim import process
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-ce7454-followup-20261017-a02')
print('TIME',datetime.now(timezone.utc).isoformat())
status=snapshot(list(range(8))); print('GPUS',json.dumps(status))
for pid in sorted({p for g in status for p in g['pids']}):
    print('PROCESS',json.dumps(process(pid)))
    print('ARGV',(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode())
for path in [Path(str(root)+'.log'),Path('/mnt/local/_gpu_guard/DISABLED'),root/'supervised/supervisor.json',root/'supervised/reclaim.json',root/'supervised/run/run.json',root/'supervised/run/study/manifest.json',root/'supervised/run/study/smoke_passed.json']:
    print('FILE',str(path),path.read_text()[-20000:] if path.exists() else 'MISSING')
study=root/'supervised/run/study'
for path in sorted(study.glob('*.log')):
    print('LOG',str(path),path.read_text()[-3500:])
for path in sorted(study.glob('*/log.csv')):
    print('EPOCHS',str(path),path.read_text()[-1200:])
for path in sorted(study.glob('numerics_*/result.json')):
    print('NUMERICS',str(path),path.read_text())
REMOTE
