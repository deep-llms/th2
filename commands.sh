#1 +30+a
#th2-q359-dot-grouped-test-20261018-a03
set -euo pipefail
TASK_PYTHON=/mnt/local/conda-py311/envs/eval_fa4/bin/python
"$TASK_PYTHON" - <<'CHECK'
import json
from pathlib import Path
p=Path('/mnt/local/_outputs/deep-llms_th2/q359-dot-grouped-20261018-a02/supervised/supervisor.json')
r=json.loads(p.read_text())
assert 'finished_at' in r and r['burn']['collective_progress_verified'] and 'handoff_error' not in r
print('PRIOR_BURN_HANDOFF_VERIFIED '+json.dumps(r),flush=True)
CHECK
"$TASK_PYTHON" scripts/bdot_b200_preflight.py --output /mnt/local/_outputs/deep-llms_th2/q359-dot-grouped-20261018-a03/preflight.json
"$TASK_PYTHON" scripts/bdot_b200_launch.py \
  --root /mnt/local/_outputs/deep-llms_th2/q359-dot-grouped-20261018-a03 \
  --session dot_grouped_20261018_a03 \
  --jobs-config resources/bdot_b200_grouped.json
