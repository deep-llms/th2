#1 +30+a
#th2-q359-ce7454-study-health-20261016-a04
set -euo pipefail
date -u
hostname
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
/mnt/local/conda-py311/envs/eval_fa4/bin/python - <<'CHECK'
from pathlib import Path
import json
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-ce7454-scratch-20261016-a04')
print('LAUNCH_LOG',Path(str(root)+'.log').read_text()[-7000:])
for name in ['supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/run.json','supervised/run/ce7454-study.log','supervised/run/study/schedule.json','supervised/burn-verified.json']:
 p=root/name
 if p.exists(): print(str(p),p.read_text()[-5000:])
for p in sorted((root/'supervised/run/study').glob('*.log')):
 print('STUDY_LOG',p.name,p.read_text()[-1800:])
for p in sorted((root/'supervised/run/study').glob('benchmark_*/result.json')):
 d=json.loads(p.read_text()); d.pop('losses',None); print('BENCHMARK',p.parent.name,d)
CHECK
