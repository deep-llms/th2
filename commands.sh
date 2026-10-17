#1 +30+a
#th2-q359-ce7454-staged-progress-20261017-a03
set -euo pipefail
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python -u - <<'REMOTE'
import json,csv
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-ce7454-staged-20261017-a01')
study=root/'supervised/run/study'
print('TIME',datetime.now(timezone.utc).isoformat())
print('GPUS',json.dumps(snapshot(list(range(8)))))
for p in [Path('/mnt/local/_gpu_guard/DISABLED'),root/'supervised/supervisor.json',root/'supervised/burn-verified.json',root/'supervised/run/run.json',study/'complete.json']:
    print('STATUS',json.dumps({'path':str(p),'data':json.loads(p.read_text()) if p.exists() else None}))
for p in sorted(study.glob('*/log.csv')):
    rows=list(csv.DictReader(p.open()))
    print('EPOCHS',json.dumps({'path':str(p),'epochs':len(rows),'last':rows[-2:]}))
for p in [Path(str(root)+'.log'),root/'supervised/run/ce7454-study.log',root/'supervised/burn.log',*sorted(study.glob('*.log'))]:
    if p.exists():print('LOG',str(p),p.read_text()[-1800:])
for p in sorted(study.glob('*_evaluation/results.json')):
    if '_best_evaluation' in str(p):continue
    r=json.loads(p.read_text())
    print('FINAL_RESULT',json.dumps({'path':str(p),'data':r}))
    for split in ('dev','test_official'):
        pred=p.parent/'external'/r['external_predictions']['splits'][split]['file']
        print('FINAL_PREDICTIONS',json.dumps({'path':str(pred),'data':json.loads(pred.read_text())}))
REMOTE
