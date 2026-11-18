#1 +60+a
#th2-ohmv-warm-downstream-replay-trace-20261118-a01
set -euo pipefail
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PYCODE'
import pathlib,json
from scripts.gpu_process import inspect
b=pathlib.Path('/mnt/local/_outputs/deep-llms_th2/warm_downstream_20261118_a02')
def read(p):
 q=b/p;return q.read_text()[-10000:] if q.exists() else None
logs={str(p.relative_to(b)):p.read_text()[-7000:] for p in list(b.glob('replay-*.log'))+list(b.glob('runs/*/raw.json.log'))}
print('REPLAY_TRACE',json.dumps(dict(logs=logs,status=read('status.json'),terminal=read('terminal.json'),failure=read('failure.json'),ownership=inspect(),guard=pathlib.Path('/mnt/local/_gpu_guard/DISABLED').exists())))
PYCODE
