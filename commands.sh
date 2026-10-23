#1 +60+a
#th2-6y3o-dot-stop-eval-inputs-20261023-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
cd /mnt/local/_outputs/deep-llms_th2/6y3o-dot-stop-adapt-20261023-a01/source
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'PYREMOTE'
import json,hashlib
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
from dataclasses import replace
from transformers import AutoTokenizer
from train import DataArguments,prepare_validation_data
from training_data import prepare_training_data,require_resume_data
from eval.stop_study import long_fragments
parent=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-long-pair-20261023-a01')
r=json.loads((parent/'prepared/D-s0.train.json').read_text())
t=AutoTokenizer.from_pretrained(r['tokenizer_name'],local_files_only=True)
a=DataArguments(**{k:v for k,v in r.items() if k in DataArguments.__dataclass_fields__})
s=SimpleNamespace(data_seed=0,seed=0,main_process_first=lambda **kw:nullcontext())
ds,identity=prepare_training_data(a,s,t,r['tokenizer_name'],1024,'fa4')
require_resume_data(parent/'prepared/runs/D-s0/checkpoint-10000',identity)
start=11000*512
cal=ds.select(range(start+256,start+512));fragments=long_fragments(cal)
val,_=prepare_validation_data(replace(a,eval_max_sequences=None),s,t,r['tokenizer_name'],1024,'fa4',identity)
assert len(val)==9704
result=dict(status='ok',raw_tokenizer_pad=t.pad_token,training_fingerprint=identity['shuffled_fingerprint'],head_fit_range=[start,start+256],calibration_range=[start+256,start+512],num_long_fragments=len(fragments),timing_tokens_sha256=hashlib.sha256(json.dumps(fragments).encode()).hexdigest(),validation_sequences=len(val))
out=Path('../evaluation-input-check.json');out.write_text(json.dumps(result,indent=2)+'\n')
print('STOP_EVAL_INPUTS '+json.dumps(result),flush=True)
PYREMOTE
