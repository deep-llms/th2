#1 +30+a
#th2-ohmv-a2-dot-data-health-20261118-a06
set -euo pipefail
date -u
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
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
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PYSTAGE'
import json
from pathlib import Path
out=Path('/mnt/local/_outputs/deep-llms_th2/a2_dot_data_20261118_a01')
root=Path('/mnt/local/_data/deep-llms_th2/a2_dot_20261118')
for name in ('dclm_packed_verified.json','fineweb_edu_packed_verified.json'):
 p=out/name
 print('PACKED_VERIFICATION',name,p.read_text() if p.is_file() else 'pending')
for corpus in ('dclm','fineweb_edu'):
 p=root/'corpora'/corpus/'status.json'
 print('CORPUS_STATUS',corpus,p.read_text() if p.is_file() else 'pending')
print('HEALTH_A04_DONE',flush=True)
PYSTAGE
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PYPROGRESS'
import json,time
from pathlib import Path
root=Path('/mnt/local/_data/deep-llms_th2/a2_dot_20261118')
for corpus in ('fineweb_edu',):
 rows=[]
 for p in sorted((root/'corpora'/corpus/'parts'/corpus).glob('*')):
  if not p.is_dir():continue
  files=[x for x in p.iterdir() if x.is_file()]
  rows.append(dict(shard=p.name,complete=(p/'complete.json').is_file(),bytes=sum(x.stat().st_size for x in files),mtime=max([x.stat().st_mtime for x in files],default=0)))
 print('PART_WRITE_PROGRESS',corpus,json.dumps(rows),flush=True)
print('HEALTH_A06_DONE',time.time(),flush=True)
PYPROGRESS
