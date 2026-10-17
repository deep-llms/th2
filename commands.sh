#1 +30+a
#th2-q359-template-realdata-preflight-20261017-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python -B - <<'PY'
import json
from pathlib import Path
from datetime import datetime,timezone
from scripts.verified_gpu_reclaim import inspect
print('TIME',datetime.now(timezone.utc).isoformat())
print('INSPECTION',json.dumps(inspect()))
cfg=json.loads(Path('deep_kv.b200.json').read_text())
for key in ('config_name','tokenizer_name','data_dir','eval_data_dir'):
 p=Path(cfg[key]);print('ASSET',key,str(p),p.exists())
 if p.is_dir():print('CONTENTS',key,[x.name for x in sorted(p.iterdir())[:8]])
 if key=='config_name':print('MODEL_CONFIG',p.read_text())
print('PREFLIGHT_COMPLETE')
PY
