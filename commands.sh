#1 +30+a
#th2-joint-v2-status-20260923-a02
set -euo pipefail
date -u
PYTHONPATH="$PWD" /usr/bin/python3 -u - <<'CHECK'
import json
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/joint-v2-b200-20260923-a03/runs')
for name in ('run.json','complete.json','report/complete.json','report/RESULTS.md','report/failure.json'):
    p=root/name
    print('FILE',name,p.read_text() if p.exists() else 'NOT_PRESENT',flush=True)
for seed in range(2):
    for arm in ('Base','Shallow','Deep'):
        directory=root/f'seed-{seed}-{arm}'
        print('ARM',directory.name,flush=True)
        for name in ('train.jsonl','validation.jsonl','complete.json','failure.json'):
            p=directory/name
            if p.exists():
                lines=p.read_text().splitlines()
                print(name,'\n'.join(lines[-2:]) if name.endswith('jsonl') else '\n'.join(lines),flush=True)
print('GUARD_DISABLED',Path('/mnt/local/_gpu_guard/DISABLED').exists(),flush=True)
print('GPU_STATUS',json.dumps(snapshot()),flush=True)
CHECK
