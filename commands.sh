#1 +15+a
#th2-q359-p7-kernel-completion-20261015-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'REMOTE'
from pathlib import Path
from datetime import datetime,timezone
import json,time
from scripts.gpu_status import snapshot
from scripts.train_then_burn import burn_progress,approved_launcher,process,BURN_HASH
from scripts.verified_gpu_reclaim import ownership
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-kernel-trial-20261015-a02')
import statistics
for attempt in range(45):
 print('POLL',datetime.now(timezone.utc).isoformat(),flush=True)
 for p in root.glob('supervised/run/timings/*/seed-42/*/step-profile-rank0.json'):
  try:
   ranks=[json.loads((p.parent/f'step-profile-rank{i}.json').read_text()) for i in range(8)]
   median=statistics.median(max(r['steps'][i]['interval_seconds'] for r in ranks) for i in range(19,49))
   print('COMPLETED_TIMING',p.parent.parent.parent.name,median,flush=True)
  except (FileNotFoundError,json.JSONDecodeError,IndexError):pass
 state=root/'supervised/supervisor.json'
 if state.exists() and json.loads(state.read_text()).get('finished_at'):break
 time.sleep(20)
print('AT',datetime.now(timezone.utc).isoformat())
for leaf in ('preflight.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/supervisor.json','supervised/run/run.json','supervised/run/gate/summary.json','supervised/run/timings/run.json','supervised/run/timing-summary.json','supervised/burn-verified.json'):
 p=root/leaf
 if p.exists():print('STATE',leaf,p.read_text())
 else:print('MISSING',leaf)
for p in root.glob('supervised/run/timings/*/seed-42/*/step-profile-rank0.json'):
 data=json.loads(p.read_text());print('PARTIAL_TIMING',str(p.relative_to(root)),json.dumps(data))
print('COMPLETE_MARKER',(root/'supervised/run/complete.json').exists())
for pattern in ('supervisor.log','supervised/run/*.log','supervised/run/timings/*.log'):
 for p in root.glob(pattern):print('LOG_TAIL',str(p.relative_to(root)),p.read_text()[-6000:])
status=snapshot(list(range(8)));print('GPU_SNAPSHOT',json.dumps(status))
b=root/'supervised/burn-verified.json';log=root/'supervised/burn.log'
if b.exists() and log.exists():
 recorded=json.loads(b.read_text());source=str(Path.cwd()/'resources/llm_pretrain_burn.py')
 workers={pid for gpu in status for pid in gpu['pids']}
 parents_ok=all(approved_launcher(process(pid)['ppid'],{source:BURN_HASH}) for pid in workers)
 print('BURN_LIVE',json.dumps(dict(ownership_matches=ownership(status)==ownership(recorded['gpus']),workers=len(workers),approved_parents=parents_ok,progress=burn_progress(log.read_text()),log_age_seconds=time.time()-log.stat().st_mtime)))
print('GUARD_DISABLED',Path('/mnt/local/_gpu_guard/DISABLED').exists())
REMOTE
