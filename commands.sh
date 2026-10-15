#1 +15+a
#th2-q359-p7-completed-eval-error-20261015-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'REMOTE'
from pathlib import Path
from datetime import datetime,timezone
import json
from scripts.train_then_burn import burn_progress
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-completed-eval-20261015-a01')
print('AT',datetime.now(timezone.utc).isoformat())
for p in sorted((root/'supervised/run/smoke').glob('*/*.log')):
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-8500));print('WORKER_LOG',p,f.read().decode(errors='replace'))
for name in ('supervised/supervisor.json','supervised/burn-verified.json'):
 p=root/name;print('STATE',name,p.read_text() if p.exists() else 'MISSING')
p=root/'supervised/burn.log';print('BURN_PROGRESS',burn_progress(p.read_text()))
REMOTE
