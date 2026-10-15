#1 +120+a
#th2-q359-p7-stop-after-keyonly-20261015-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'REMOTE'
import json
from pathlib import Path
from datetime import datetime,timezone
from deep_kv.report import report
from scripts.gpu_status import snapshot
from scripts import stop_proxy_followup as stop
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-exploration8-20261015-a03')
run=root/'supervised/run'
print('AT',datetime.now(timezone.utc).isoformat(),flush=True)
print('GPU_BEFORE',json.dumps(snapshot(list(range(8)))),flush=True)
folder=run/'lm-keyonly/seed-42/P7-simple-lm-keyonly'
p=folder/'result.json'
if not p.exists():
 print('KEYONLY_NOT_FINISHED_NO_SIGNALS',flush=True)
 raise SystemExit(0)
r=json.loads(p.read_text())
assert r['arm']=='P7-simple-lm-keyonly' and r['global_step']==2500 and r['status']=='complete'
assert json.loads((run/'lm-keyonly/validated.json').read_text())['status']=='passed'
assert json.loads((folder/'checkpoint-2500/trainer_state.json').read_text())['global_step']==2500
report(folder.parent,['P7-simple-lm-keyonly'],expected_step=2500,expected_seed=42,expected_attention_backend='fa4')
print('KEYONLY_COMPLETE',json.dumps(r),flush=True)
for label in ('a-long','targetgrad-long','lm-long','lm-sparse','lm-keyonly','lm-valueonly','lm-noaux','lm-nexttoken'):
 for p in (run/label/'seed-42').glob('*/result.json'):
  value=json.loads(p.read_text());print('COMPLETED_RESULT',label,json.dumps(value),flush=True)
stop.ROOT=root;stop.SESSION=root.name;stop.HOST='thiennh-p6-q359-worker-0';stop.BURN_SESSION=root.name+'-final-burn'
stop.stop(root/'user-stop-after-keyonly.json')
print('FINAL_SUPERVISOR',(root/'supervised/supervisor.json').read_text(),flush=True)
REMOTE
