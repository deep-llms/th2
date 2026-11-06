#1 +20+a
#th2-wc69-family-mixture-schema-20261107-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone
import pyarrow.parquet as pq
plan=json.loads(Path('resources/a2_family_mixture_plan_20261107.json').read_text());schemas=[]
for entry in plan['sources']:
 for record in entry['files'][:1]:
  p=Path(record['local'])
  if p.is_dir():p=p/Path(record['path']).name
  row=dict(source=entry['name'],path=str(p),exists=p.is_file(),expected_bytes=record['bytes'])
  if p.is_file():
   row['bytes']=p.stat().st_size
   if p.suffix=='.parquet':
    try:row['schema']=str(pq.ParquetFile(p).schema_arrow)
    except Exception as e:row['error']=repr(e)
  schemas.append(row)
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-mixture-20261107-a01/mixture')
def artifact(p):
 if not p.is_file():return None
 raw=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
r=dict(utc=datetime.now(timezone.utc).isoformat(),schemas=schemas,progress=artifact(root/'progress.json'),complete=artifact(root/'complete.json'),sources={p.parent.name:artifact(p) for p in root.glob('sources/*/complete.json')},session=subprocess.run(['tmux','display-message','-p','-t','a2_family_mix_20261107_a01','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip())
print('FAMILY_MIXTURE_SCHEMA',json.dumps(r),flush=True)
PY
