#1 +60+a
#th2-q359-p7-entries-completion-20261015-a01
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
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-entries-integrated-20261015-a01')
deadline=time.monotonic()+300
while time.monotonic()<deadline:
 p=root/'supervised/supervisor.json'
 state=json.loads(p.read_text()) if p.exists() else {}
 if state.get('finished_at'):break
 print('WAIT_FOR_COMPLETION',datetime.now(timezone.utc).isoformat(),state.get('training_status'),flush=True)
 time.sleep(15)
print('AT',datetime.now(timezone.utc).isoformat())
previous=root.parent/'q359-p7-fusion-trial-20261015-a01'
for leaf in ('supervised/run/timing-summary.json','supervised/supervisor.json','supervised/burn-verified.json'):
 p=previous/leaf
 if p.exists():print('PREVIOUS',leaf,p.read_text())

for leaf in ('preflight.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/supervisor.json','supervised/run/run.json','supervised/run/gate/summary.json','supervised/run/summary.json','supervised/burn-verified.json'):
 p=root/leaf
 if p.exists():print('STATE',leaf,p.read_text())
 else:print('MISSING',leaf)
print('COMPLETE_MARKER',(root/'supervised/run/complete.json').exists())
for pattern in ('supervisor.log','supervised/run/*.log'):
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
