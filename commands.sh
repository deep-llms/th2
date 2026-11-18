#1 +30+a
#th2-ohmv-warm-downstream-inventory-20261118-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PYCODE'
import json,pathlib,sys
from scripts.gpu_process import inspect
from efficient_llm.warm_order.data import sha
roots=[pathlib.Path('/mnt/local/_data/deep-llms_th2'),pathlib.Path('/mnt/local/_outputs/deep-llms_th2')]
print('INVENTORY',json.dumps({'ownership':inspect(),'benchmarks':[str(p) for r in roots for p in r.glob('**/*') if p.is_file() and (p.suffix=='.parquet' and any(t in str(p).lower() for t in ['hellaswag','piqa','arc-easy','lambada']) or p.name=='requests.json')], 'tokenizer':{p.name:sha(p) for p in pathlib.Path('/mnt/local/_data/deep-llms_th2/a2_dot_20261118/tokenizer').glob('*.json')}}))
PYCODE
CUDA_VISIBLE_DEVICES='' /mnt/local/conda-py311/envs/eval/bin/python3.11 - <<'PYCODE'
from importlib.metadata import version
import torch,transformers,lm_eval,pyarrow
print('EVAL_RUNTIME', {p:version(p) for p in ('torch','transformers','lm-eval','datasets','tokenizers','numpy')})
print('INVENTORY_DONE')
PYCODE
