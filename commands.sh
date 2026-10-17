#1 +30+a
#th2-q359-ce7454-soups-final-review-20261017-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python -u - <<'REMOTE'
import json
from pathlib import Path
from datetime import datetime,timezone
from scripts.verified_gpu_reclaim import inspect
print('TIME',datetime.now(timezone.utc).isoformat())
print('INSPECTION',json.dumps(inspect()))
for name in ('q359-ce7454-staged-20261017-a01','q359-ce7454-soups-20261017-a01'):
    root=Path('/mnt/local/_outputs/deep-llms_th2')/name
    study=root/'supervised/run/study'
    for p in (root/'supervised/supervisor.json',root/'supervised/burn-verified.json',root/'supervised/run/run.json',study/'complete.json'):
        print('STATUS',json.dumps({'path':str(p),'data':json.loads(p.read_text()) if p.exists() else None}))
    for p in (Path(str(root)+'.log'),root/'supervised/run/ce7454-study.log',root/'supervised/burn.log'):
        if p.exists():
            lines=p.read_text().splitlines()
            print('LOG',str(p),'\n'.join(lines[:12]+lines[-35:]))
    for p in sorted(study.glob('*_evaluation/results.json')):
        if '_best_evaluation' in str(p):continue
        r=json.loads(p.read_text())
        print('FINAL_RESULT',json.dumps({'path':str(p),'data':r}))
        for split in ('dev','test_official'):
            pred=p.parent/'external'/r['external_predictions']['splits'][split]['file']
            print('FINAL_PREDICTIONS',json.dumps({'path':str(pred),'data':json.loads(pred.read_text())}))
REMOTE
