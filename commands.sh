#1 +30+a
#th2-ohmv-a2-dot-data-health-20261118-a03
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
TASK_SOURCE=/mnt/local/_outputs/deep-llms_th2/a2_dot_data_20261118_a01/source
cd "$TASK_SOURCE"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.a2_dot_rebuild_data health
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PYHEALTH'
import hashlib,json,subprocess
from pathlib import Path
out=Path('/mnt/local/_outputs/deep-llms_th2/a2_dot_data_20261118_a01')
source=json.loads((out/'source-manifest.json').read_text());assert all(hashlib.sha256((out/'source'/p).read_bytes()).hexdigest()==h for p,h in source['files'].items())
root=Path('/mnt/local/_data/deep-llms_th2/a2_dot_20261118')
parts={c:[p.parent.name for p in sorted((root/'corpora'/c/'parts'/c).glob('*/complete.json'))] for c in ['dclm','fineweb_edu']}
print('FROZEN_SOURCE_PASS',len(source['files']))
print('HASH_ADMITTED_TOKENIZED_SHARDS',json.dumps(parts))
print('TMUX_CPU_ALIVE',subprocess.run(['tmux','has-session','-t','a2_dot_data_20261118_a01'],capture_output=True).returncode==0)
assert not (out/'failure.json').exists(), 'CPU worker failed; inspect health output'
PYHEALTH
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PYFILES'
import hashlib,json,os
from pathlib import Path
from scripts.a2_dot_rebuild_data import recipe,plan_for
cfg=recipe();records=plan_for(cfg,'dclm')['corpora']['dclm']['files'];result=[]
for rec in records[40:50]:
 p=Path(rec['local']);p=p/Path(rec['path']).name if p.is_dir() else p
 row={'index':rec['index'],'exists':p.is_file(),'bytes':p.stat().st_size if p.is_file() else None,'expected_bytes':rec['bytes']}
 if p.is_file() and p.stat().st_size==rec['bytes']:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8<<20),b''):h.update(block)
  row.update(sha256=h.hexdigest(),sha_pass=h.hexdigest()==rec['sha256'])
 result.append(row)
print('BATCH05_ACTUAL_FILES',json.dumps(result),flush=True)
procs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cmd=(p/'cmdline').read_bytes()
  if b'a2_dot_20261118' in cmd or b'a2_dot_rebuild_data' in cmd:
   procs.append({'pid':int(p.name),'comm':(p/'comm').read_text().strip(),'cmd_sha256':hashlib.sha256(cmd).hexdigest()})
 except (OSError,ProcessLookupError):pass
print('RELATED_PROCESS_IDENTITIES',json.dumps(procs),flush=True)
PYFILES
