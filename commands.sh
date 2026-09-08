#1 +30+a
#th2-swt-identify-a03-before-authorized-cancel-clean-20260908-a01
set -euo pipefail
date -u
hostname
/mnt/local/conda-py311/envs/swt/bin/python - <<'PY'
import json
from pathlib import Path
from scripts.reclaim_verified_burn import identity
from scripts.gpu_status import snapshot
root = '/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_10k_s42_20260908_a03'
expected = ['bash','scripts/train_capacity_b200.sh',root,'B0','A128','A256','A512','C','D']
records = {}
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) <= 1:
        continue
    try:
        records[int(proc.name)] = identity(int(proc.name))
    except (FileNotFoundError, ProcessLookupError):
        pass
queues = [r for r in records.values() if r['argv'] == expected]
print('EXACT_QUEUES', json.dumps(queues), flush=True)
for queue in queues:
    owned = {queue['pid']}
    while True:
        expanded = owned | {p for p,r in records.items() if r['parent'] in owned}
        if expanded == owned:
            break
        owned = expanded
    print('OWNED_DESCENDANTS', json.dumps([records[p] for p in sorted(owned)]), flush=True)
    if queue['parent'] in records:
        print('PROTECTED_PARENT', json.dumps(records[queue['parent']]), flush=True)
print('GPU_STATUS', json.dumps(snapshot()), flush=True)
for name in (root, '/mnt/local/_data/deep-llms_th2/swt/qwen_en_map160_batch1000'):
    p = Path(name)
    print('TARGET', name, 'exists', p.exists(), 'resolved', str(p.resolve()), flush=True)
    if p.is_dir():
        print('CHILDREN', sorted(x.name for x in p.iterdir())[:20], flush=True)
log = Path(root)/'pipeline.log'
if log.exists():
    with log.open('rb') as f:
        f.seek(max(0, log.stat().st_size-3500))
        print('PIPELINE_TAIL', f.read().decode(errors='replace'), flush=True)
print('SWT_A03_READONLY_ANCESTRY_COMPLETE', flush=True)
PY
