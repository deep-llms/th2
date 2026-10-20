#1 +30+a
#th2-6y3o-dot-N-preflight-20261020-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
export CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'PY'
from pathlib import Path
import datetime,json,shutil
from scripts.native_baseline_pipeline import burn_gate
from scripts.gpu_process import inspect
from scripts.train_then_burn import burn_progress,digest,GUARD_HASH,BURN_HASH
old=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-pilot-20261020-a01')
new=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-N-20261020-a01')
burn_gate(old)
assert burn_progress((old/'supervised/burn.log').read_text())
assert json.loads((old/'supervised/run/run.json').read_text())['status']=='ok'
assert not new.exists() and not Path(str(new)+'.prepare.log').exists()
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
assert digest('resources/llm_pretrain_burn.py')==BURN_HASH
assert shutil.disk_usage('/mnt/local').free > 200*1024**3
print('OBSERVED_UTC',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
print('GPU_LIVE',json.dumps(inspect()),flush=True)
print('DISK_BYTES',json.dumps(shutil.disk_usage('/mnt/local')._asdict()),flush=True)
for name in ['prepared/P1-A.train.json','prepared/P1-A.model.json','assets-verified.json','cache-ready.json','prepared/validation.json','supervised/run/T12/summary.json','supervised/run/T14/summary.json']:
 p=old/name;j=json.loads(p.read_text())
 if name=='assets-verified.json':j={k:j[k] for k in ['status','files_verified','data_root','subset_root','tokenizer','contract']}
 if name=='cache-ready.json':j={k:j[k] for k in ['status','train_sequences','validation_sequences']}
 if name=='prepared/validation.json':j={'sha256':digest(p),'periodic_sequences':len(j['periodic_indices']),'final_sequences':len(j['final_indices'])}
 if name.endswith('T12/summary.json'):j={k:v for k,v in j.items() if k!='results'}
 print('PREVIOUS',name,json.dumps(j),flush=True)
print('N_PREFLIGHT_PASSED',flush=True)
PY
