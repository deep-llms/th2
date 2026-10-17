#1 +30+a
#th2-q359-ce7454-followup-review-20261017-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python -u - <<'REMOTE'
import json
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-ce7454-followup-20261017-a02')
study=root/'supervised/run/study'
print('TIME',datetime.now(timezone.utc).isoformat())
print('GPUS',json.dumps(snapshot(list(range(8)))))
for p in [Path('/mnt/local/_gpu_guard/DISABLED'),root/'supervised/supervisor.json',root/'supervised/burn-verified.json',root/'supervised/run/run.json',study/'complete.json']:
    print('STATUS',json.dumps({'path':str(p),'data':json.loads(p.read_text()) if p.exists() else None}))
for p in [Path(str(root)+'.log'),root/'supervised/run/ce7454-study.log',root/'supervised/burn.log']:
    if p.exists():print('LOG',str(p),p.read_text()[-16000:])
for name in ['grip_nodown_ibna_s1','grip_nodown_ibnb_s1','grip_nodown_s1','srresnet_s1','srresnet_stats_s1']:
    for p in [study/(name+'.log'),study/name/'log.csv',study/(name+'_evaluation.log')]:
        print('ARM_LOG',str(p),p.read_text()[-2200:] if p.exists() else 'MISSING')
    p=study/(name+'_evaluation')/'results.json'
    if p.exists():
        r=json.loads(p.read_text()); print('FINAL_RESULT',json.dumps({'path':str(p),'data':r}))
        for s in ('dev','test_official'):
            pred=p.parent/'external'/r['external_predictions']['splits'][s]['file']
            print('FINAL_PREDICTIONS',json.dumps({'path':str(pred),'data':json.loads(pred.read_text())}))
REMOTE
