#1 +60+a
#th2-tjx3-fa4-dense-numerical-review-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/train_env/bin/python -u - <<'CHECK'
import json,hashlib,datetime
from pathlib import Path
print('READ_ONLY_REVIEW',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
root=Path('/mnt/local/_outputs/deep-llms_th2')
for backend,folder in [('sdpa','proxy-baseline-A-2500-20261005-a01'),('fa4','proxy-baseline-A-fa4-2500-20261005-a01')]:
    run=root/folder/'supervised/run';arm=run/'baseline/seed-42/A'
    names=['train_config.json','trainer_state.json','result.json']
    result=json.loads((arm/'result.json').read_text())
    names+=result.get('sdpa_receipts',[])+result.get('fa4_receipts',[])
    for name in names:
        path=arm/name;raw=path.read_bytes()
        print('REVIEW_ARTIFACT',json.dumps(dict(backend=backend,name=name,sha256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))),flush=True)
    if backend=='fa4':
        for name in ['numerics.json','baseline/validation.json']:
            raw=(run/name).read_bytes()
            print('REVIEW_ARTIFACT',json.dumps(dict(backend=backend,name=name,sha256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))),flush=True)
print('READ_ONLY_REVIEW_COMPLETE',flush=True)
CHECK
