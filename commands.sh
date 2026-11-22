#1 +120+a
#th2-ohmv-attention-shift-cpu-20261122-a02
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH="$PWD:$PWD/tests/warm_order"
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
"$TASK_PYTHON" - <<'PY'
import json,shutil,socket
from pathlib import Path
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect
import torch,numpy
assert socket.gethostname()=='thiennh-p6-ohmv-worker-0'
assert torch.__version__=='2.14.1+cu130' and numpy.__version__=='2.4.6'
p=Path('resources/attention_shift_source_20261121.json');assert sha(p)=='a72c613e71b90bc3a7d97022bd850af2f3804823f66bc35826b4bf68d9687fe3'
pins=json.loads(p.read_text())['files'];base=Path('/mnt/local/_outputs/deep-llms_th2/attention_shift_20261121_a02')
assert not base.exists(),'Existing output; reconcile before any retry'
for name,h in pins.items():assert sha(name)==h,name
base.mkdir(parents=True)
for name,h in pins.items():
 dest=base/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest);assert sha(dest)==h
(base/'source.json').write_text(json.dumps(pins,indent=2)+'\n')
(base/'cpu-preownership.json').write_text(json.dumps(inspect(),indent=2)+'\n')
print('ATTENTION_SHIFT_CPU_START',json.dumps(dict(source_pins=len(pins))),flush=True)
PY
cd /mnt/local/_outputs/deep-llms_th2/attention_shift_20261121_a02/source
export PYTHONPATH="$PWD:$PWD/tests/warm_order"
"$TASK_PYTHON" -m unittest test_attention_shift.AttentionShift test_attention_shift_eval.ShiftEvaluation test_attention_shift_node.ShiftNode -v
"$TASK_PYTHON" -m efficient_llm.warm_order.attention_shift_node prepare --path resources/attention_shift_trial_20261121.json --sha f56424abf74ee3755ea6a69cf526b9afa994a6d2aab1b3df882da8d6e5ea24f0 --out /mnt/local/_outputs/deep-llms_th2/attention_shift_20261121_a02/prepared
"$TASK_PYTHON" - <<'PY'
import json
from pathlib import Path
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect
base=Path('/mnt/local/_outputs/deep-llms_th2/attention_shift_20261121_a02');old=base.with_name('layer_adaptation_20261121_continuation_a01')
r=dict(prepared_sha256=sha(base/'prepared/prepared.json'),prepared=json.loads((base/'prepared/prepared.json').read_text()),current_ownership=inspect(),old_study={})
for name in ('status.json','failure.json','terminal.json','scientific-complete.json'):
 p=old/name
 if p.exists():r['old_study'][name]=json.loads(p.read_text())
p=old/'backup.json'
if p.exists():
 b=json.loads(p.read_text());r['old_backup']=dict(archive_sha256=b['archive_sha256'],archive_bytes=b['archive_bytes'],copies=len(b['node_fullstate_copies']),node_replay=b['node_replay'])
(base/'cpu-terminal.json').write_text(json.dumps(r,indent=2)+'\n')
print('ATTENTION_SHIFT_CPU_COMPLETE',json.dumps(r),flush=True)
PY
