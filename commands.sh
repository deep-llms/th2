#1 +30+a
#th2-q359-ce7454-staged-health-20261017-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python -u - <<'REMOTE'
import json
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-ce7454-staged-20261017-a01')
old=Path('/mnt/local/_outputs/deep-llms_th2/q359-ce7454-followup-20261017-a02')
study=root/'supervised/run/study'
print('TIME',datetime.now(timezone.utc).isoformat())
print('GPUS',json.dumps(snapshot(list(range(8)))))
for path in [old/'supervised/supervisor.json',old/'supervised/burn-verified.json',Path('/mnt/local/_gpu_guard/DISABLED'),root/'supervised/supervisor.json',root/'supervised/gpus-free-before-training.json',root/'supervised/run/run.json',study/'manifest.json',study/'best_evaluations_complete.json',study/'complete.json']:
    print('STATUS',json.dumps({'path':str(path),'data':json.loads(path.read_text()) if path.exists() else None}))
for path in [Path(str(root)+'.log'),root/'supervised/run/ce7454-study.log',old/'supervised/burn.log',*sorted(study.glob('*.log')),*sorted(study.glob('*/log.csv')),*sorted(study.glob('*smoke_passed.json'))]:
    if path.exists():print('LOG',str(path),path.read_text()[-7000:])
for path in [old/'supervised/run/study/grip_nodown_ibna_s1_evaluation/results.json',*sorted(study.glob('*_evaluation/results.json'))]:
    if path.exists():
        r=json.loads(path.read_text());print('FINAL_RESULT',json.dumps({'path':str(path),'data':r}))
        for split in ('dev','test_official'):
            p=path.parent/'external'/r['external_predictions']['splits'][split]['file']
            print('FINAL_PREDICTIONS',json.dumps({'path':str(p),'data':json.loads(p.read_text())}))
REMOTE
