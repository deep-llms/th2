#1 +30+a
#th2-6y3o-dot-downstream-preflight-20261023-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/eval/bin/python3.11 - <<'REMOTE'
import json, importlib.metadata as m, hashlib, subprocess
from pathlib import Path
from scripts.gpu_process import inspect
from scripts.native_baseline_pipeline import validate_burn_handoff
from scripts.train_then_burn import burn_progress
import lm_eval
from lm_eval.models.huggingface import HFLM
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-long-pair-20261023-a01')
read=lambda p:json.loads(p.read_text())
live=inspect();state=read(root/'supervised/supervisor.json');proof=read(root/'supervised/burn-verified.json')
validate_burn_handoff(state,proof,live)
assert burn_progress((root/'supervised/burn.log').read_text())
versions={k:m.version(k) for k in ['torch','transformers','datasets','accelerate','lm-eval','flash-attn-4','tokenizers','safetensors']}
data=Path('/mnt/local/_data/deep-llms_th2')
folders=[str(p) for p in data.iterdir() if p.is_dir()]
target=data/'benchmarks_zero_20261023';manifest=read(Path('resources/downstream_english_20261023.json'))
files={f['path']:(target/f['path']).is_file() for r in manifest['repositories'] for f in r['files']}
print('DOWNSTREAM_PREFLIGHT '+json.dumps(dict(live=live,versions=versions,folders=folders,dataset_files=files,previous_report=read(root/'supervised/run/long-pair-report.json'))),flush=True)
REMOTE
