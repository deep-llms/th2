#1 +15+a
#th2-q359-sparse-checkpoint-inventory-20261013-a01
set -euo pipefail
date -u
hostname
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
python3 - <<'PYREMOTE'
import os,json
from pathlib import Path
for root in ['/mnt/local/_outputs/deep-llms_th2','/mnt/local/_models/deep-llms_th2']:
    p=Path(root)
    print('ROOT',root,'exists',p.exists(),flush=True)
    if not p.exists():continue
    print('TOP_LEVEL',json.dumps(sorted(x.name for x in p.iterdir())),flush=True)
    matches=[]
    for current,dirs,files in os.walk(p):
        dirs[:]=[d for d in dirs if d not in ('wandb','.git','cache','datasets','logs')]
        weights=[f for f in files if f.endswith(('.safetensors','.bin')) and ('model' in f or f=='proxy.bin')]
        if weights:matches.append(dict(path=current,files=[dict(name=f,bytes=os.path.getsize(os.path.join(current,f))) for f in weights]))
    print('WEIGHT_INVENTORY',json.dumps(matches),flush=True)
print('INVENTORY_COMPLETE',flush=True)
PYREMOTE
