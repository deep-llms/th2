#1 +30+a
#th2-wc69-dot-pool-sr-correction-terminal-20261112-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
TASK_PYTHON=/mnt/local/conda-py311/envs/dot_pool/bin/python3.11
"$TASK_PYTHON" -u - <<'PY'
import json,time,hashlib
from pathlib import Path
from scripts.gpu_process import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/dot_pool_20261112/sr-recheck-a01')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
files={}
for name in ['submitted.json','admission.json','source-manifest.json','supervised/supervisor.json','supervised/run/run.json','supervised/run/complete.json','supervised/burn-verified.json']:
 p=root/name
 if p.is_file():files[name]={'sha256':sha(p),'value':json.loads(p.read_text())}
manifest=json.loads((root/'source-manifest.json').read_text())
source_errors=[r for r,h in manifest['files'].items() if sha(root/'source'/r)!=h]
runs={}
for seed in [0,1]:
 p=root/'supervised/run'/f'E1-SR-s{seed}';row={}
 for name in ['started.json','verification.json','complete.json']:
  f=p/name
  if f.is_file():row[name]={'sha256':sha(f),'value':json.loads(f.read_text())}
 if (p/'complete.json').is_file():
  complete=json.loads((p/'complete.json').read_text());archive=Path(complete['path'])
  row['archive_matches']=archive.stat().st_size==complete['bytes'] and sha(archive)==complete['sha256']
 runs[p.name]=row
input_errors=[]
for seed in [0,1]:
 expected=json.loads((root/'source'/f'expected-s{seed}.json').read_text())
 for name,rec in expected['inputs'].items():
  p=Path(name)
  if p.stat().st_size!=rec['bytes'] or sha(p)!=rec['sha256']:input_errors.append(name)
 manifest_path=Path(expected['data_manifest'])
 if sha(manifest_path)!=expected['data_manifest_sha256']:input_errors.append(str(manifest_path))
 for name,rec in json.loads(manifest_path.read_text())['files'].items():
  p=manifest_path.parent/name
  if p.stat().st_size!=rec['bytes'] or sha(p)!=rec['sha256']:input_errors.append(str(p))
logs={}
for name in ['supervisor.log','supervised/burn.log','supervised/run/E1-SR-s0-diagnostics.log','supervised/run/E1-SR-s1-diagnostics.log']:
 p=root/name
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-5000));logs[name]=f.read().decode(errors='replace')
print('DOT_SR_TERMINAL_JSON',json.dumps({'utc':time.time(),'files':files,'source_errors':source_errors,'input_errors':input_errors,'runs':runs,'logs_tail':logs,'ownership':inspect()}),flush=True)
PY
