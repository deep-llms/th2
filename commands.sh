#1 +20+a
#th2-wc69-a2-why-input-inventory-20261104-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect
print('UTC',datetime.now(timezone.utc).isoformat());print('GPU',json.dumps(inspect()))
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-pause-20261104-a01')
for name in ('pause/request.json','pause/saved_state.json','pause/signal.json','pause/terminal.json'):
 p=root/name
 if p.exists():print('PAUSE',name,p.read_text())
print('PAUSE_LOG_TAIL',(root/'pause.log').read_text()[-4000:])
print('DISK',shutil.disk_usage('/mnt/local'))
plan=json.loads(Path('resources/a2_corpora_30b_20261029.json').read_text())
items=plan['corpora']['fineweb_edu']['files'][:44]
print('FWE_RAW',json.dumps([dict(path=x['local'],exists=Path(x['local']).exists(),bytes=Path(x['local']).stat().st_size if Path(x['local']).exists() else None) for x in items]))
assets=json.loads(Path('resources/a2_inputs_20261028.json').read_text())['assets']
for key in ('fwe/P300','fwe/P1T'):
 entry=assets[key]
 for dirname,hashes in ((entry['weights_dir'],entry['files']),(entry['code_dir'],entry['code_files'])):
  for name,expected in hashes.items():
   p=Path(dirname)/name;b=Path('/mnt/local/_backup/deep-llms_th2')/p.relative_to('/mnt/local/_models/deep-llms_th2')
   print('ASSET',json.dumps(dict(key=key,path=str(p),exists=p.is_file(),backup=str(b),backup_exists=b.is_file(),expected_sha256=expected)))
for p in (Path('/mnt/local/_data/deep-llms_th2/a2_corpora_30b_20261029'),Path('/mnt/local/_data/deep-llms_th2/a2_stage1b_fwe_shuffle2030_20261030')):
 print('DATA_PATH',str(p),p.exists())
 if p.exists():print('TOP_CONTENTS',[x.name for x in p.iterdir()])
PY
