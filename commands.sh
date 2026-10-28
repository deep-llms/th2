#1 +30+a
#th2-6y3o-a2-corpus-inventory-20261029-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'INVENTORY'
import json,os,socket,shutil
from pathlib import Path
from importlib.metadata import version
from datetime import datetime,timezone
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
print('CORPUS_INVENTORY',datetime.now(timezone.utc).isoformat(),flush=True)
print('CPU',os.cpu_count(),len(os.sched_getaffinity(0)),flush=True)
for p in ['/mnt/local','/mnt/local/_data/deep-llms_th2','/mnt/local/_outputs/deep-llms_th2']:
 print('DISK',p,shutil.disk_usage(p),flush=True)
for n in ['pyarrow','tokenizers','numpy','datasets']:
 print('RUNTIME',n,version(n),flush=True)
for p in Path('/mnt/local/_data/deep-llms_th2').glob('*'):
 print('DATA_ENTRY',str(p),flush=True)
print('TOKENIZER',Path('/mnt/local/_models/deep-llms_th2/a2/dclm/P300/tokenizer.json').stat().st_size,flush=True)
print('INVENTORY_COMPLETE',flush=True)
INVENTORY
