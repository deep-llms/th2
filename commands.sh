#1 +30+a
#th2-ohmv-a2-dot-data-health-20261118-a01
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
