#1 +30+a
#th2-wc69-a2-data-rebuild-20261103-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
"$TASK_PYTHON" -u -m scripts.a2_rebuild_wc69_data --code-commit 4900d756b61c24358eb9e81d308147275f4d2c3a
sleep 20
"$TASK_PYTHON" - <<'PY'
from pathlib import Path
import datetime,json
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-data-rebuild-20261103-a02')
print('SNAPSHOT',datetime.datetime.now(datetime.timezone.utc).isoformat())
for name in ('source-manifest.json','restore.json','failure.json','complete.json'):
 p=root/name
 if p.exists(): print(name,p.read_text())
p=root/'prepare.log'
if p.exists():print('LOG_TAIL',p.read_text()[-14000:])
if (root/'failure.json').exists():raise SystemExit(1)
PY
