#1 +30+a
#th2-q359-ce7454-watch-review-20261016-a07
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
export CUDA_VISIBLE_DEVICES=''
python3 - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json
base=Path('/mnt/local/_outputs/deep-llms_th2')
original=base/'q359-ce7454-scratch-20261016-a04'
band=base/'q359-ce7454-band-20261016-a01'
watch=base/'q359-ce7454-interim-watch-20261016-a01'
print('REMOTE_UTC',datetime.now(timezone.utc).isoformat())
for path in [original/'supervised/run/run.json',original/'supervised/supervisor.json',band/'controller.json',watch/'watch/owner.json',watch/'watch/latest_status.json',band/'released.json',watch/'watch/released.json',original/'supervised/burn-verified.json']:
 print('STATE',str(path),path.read_text() if path.exists() else 'absent')
for root in [original/'supervised/run/study',band/'study']:
 print('SCHEDULE',str(root),(root/'schedule.json').read_text())
 for path in sorted(root.glob('*/log.csv')):
  if path.parent.name.startswith('entrypoint_'):continue
  lines=path.read_text().splitlines()
  print('LIVE_PROGRESS',json.dumps({'path':str(path),'rows':lines[-3:]}))
 for path in sorted(root.glob('*_evaluation/results.json')):
  print('FINAL_RESULT',json.dumps({'path':str(path),'data':json.loads(path.read_text())}))
  for pred in sorted((path.parent/'external').glob('*predictions.json')):
   print('FINAL_PREDICTIONS',json.dumps({'path':str(pred),'data':json.loads(pred.read_text())}))
 for elog in sorted(root.glob('*_evaluation.log')):
  print('EVAL_LOG_TAIL',str(elog),'\n'.join(elog.read_text().splitlines()[-8:]))
 print('STUDY_COMPLETE',str(root),(root/'complete.json').read_text() if (root/'complete.json').exists() else 'absent')
guard=Path('/mnt/local/_gpu_guard/DISABLED')
print('GPU_GUARD',guard.read_text() if guard.is_file() else str(guard.exists()))
print((watch/'watch.log').read_text())
PY
ps -p 1236924,1271652 -o pid,lstart,stat,args
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv
