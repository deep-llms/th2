#1 +60+a
#th2-q359-p6-compile-results-20261011-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import json,socket,hashlib
from datetime import datetime,timezone
from pathlib import Path
from scripts.gpu_status import snapshot
from scripts.train_then_burn import digest
assert socket.gethostname()=='thiennh-p6-q359-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p6-compile-20261011-a02');run=root/'supervised/run'
for relative in ('numerics/summary.json','summary.json'):
    p=run/relative
    print('ARTIFACT_JSON',relative,digest(p),p.read_text(),flush=True)
for name in ('A-first','P6-eager-first','P6-compiled-first','P6-compiled-second','P6-eager-second','A-second'):
    arm='A' if name.startswith('A-') else 'P6-iso'
    r=json.loads((run/name/'seed-1042'/arm/'train_config.json').read_text())
    print('RECIPE',json.dumps(dict(name=name,arm=r['pilot']['arm'],seed=r['training']['seed'],data_seed=r['training']['data_seed'],
        proxy_module_seed=r['pilot'].get('proxy_module_seed'),world_size=r['world_size'],tokens_per_update=r['tokens_per_update'],
        train_fingerprint=r['train_fingerprint'],eval_fingerprint=r['eval_fingerprint'])),flush=True)
checkpoint=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-10k-seed1042-20261009-a01/supervised/run/seed-1042/P6-iso/checkpoint-10000/model.safetensors')
assert digest(checkpoint)=='3e04cc0a27f1e0d5a1e97efdc1e34c11d4e821f9b34fb76898a0652fd8d28442'
print('SOURCE_CHECKPOINT_STILL_UNCHANGED',flush=True)
print('AT',datetime.now(timezone.utc).isoformat(),flush=True)
print('SUPERVISOR',(root/'supervised/supervisor.json').read_text(),flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
p=root/'supervised/burn.log'
with p.open('rb') as f:
    f.seek(max(0,p.stat().st_size-15000));data=f.read().decode(errors='replace')
print('BURN_PROGRESS','\n'.join(data.splitlines()[-8:]),flush=True)
PYREMOTE
