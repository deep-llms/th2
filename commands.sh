#1 +30+a
#th2-6y3o-efficient-kv-long-preflight-20261027-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 -u - <<'READONLY'
import json, shutil, socket
from pathlib import Path
from datetime import datetime, timezone
from scripts.gpu_process import inspect
assert socket.gethostname() == 'thiennh-p6-6y3o-worker-0'
root = Path('/mnt/local/_outputs/deep-llms_th2')
print('HEALTH_AT', datetime.now(timezone.utc).isoformat(), flush=True)
print('GPU_INSPECTION', json.dumps(inspect()), flush=True)
print('FREE_BYTES', shutil.disk_usage(root).free, flush=True)
data = Path('/mnt/local/_data/deep-llms_th2/raw/cx_sampled_dd1b1513')
contract = json.loads(Path('resources/cx_english_20261020.json').read_text())
pool = json.loads((data/'subsets'/contract['subset']/'manifest.json').read_text())
print('TRAIN_SHARDS', len(pool['shards']['train']), flush=True)
print('FIRST_SHARD', json.dumps(pool['shards']['train'][0]), flush=True)
prior = root/'6y3o-efficient-kv-qkv-20261027-a01'
for rel in ['supervised/run/complete.json', 'supervised/supervisor.json', 'supervised/burn-verified.json']:
 print('PRIOR', rel, (prior/rel).read_text(), flush=True)
print('HEALTH_COMPLETE', flush=True)
READONLY
# KV recovery study complete; verified results and communicating GPU burns.
