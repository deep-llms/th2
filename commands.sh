#1 +30+a
#th2-ohmv-warm-downstream-data-proof-20261118-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
CUDA_VISIBLE_DEVICES='' /mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m unittest tests.warm_order.test_downstream -v
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PYCODE'
import json,pathlib
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect
m=json.loads(pathlib.Path('resources/warm_downstream_data_20261118.json').read_text());results=[]
for repo in m['repositories']:
 for f in repo['files']:
  p=pathlib.Path('/mnt/local/_data/deep-llms_th2/warm_downstream_20261118')/f['path'];actual=sha(p) if p.is_file() else None
  results.append(dict(path=str(p),sha256=actual,ok=actual==f['sha256'] and p.stat().st_size==f['bytes']))
pin=json.loads(pathlib.Path('resources/warm_downstream_source_20261118.json').read_text())
errors=[p for p,w in pin['files'].items() if sha(p)!=w]
print('DOWNSTREAM_DATA_PROOF',json.dumps(dict(files=results,all_ok=all(r['ok'] for r in results),source_errors=errors,ownership=inspect())))
PYCODE
