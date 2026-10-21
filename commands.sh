#1 +30+a
#th2-6y3o-dot-diagnostics-preflight-20261021-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline OMP_NUM_THREADS=1 /mnt/local/conda-py311/envs/train/bin/python3.11 - <<'PY'
import json,shutil,socket
from pathlib import Path
from scripts.gpu_process import inspect
from scripts.native_baseline_pipeline import burn_gate
from scripts.train_then_burn import burn_progress,digest,GUARD_HASH,BURN_HASH
root=Path('/mnt/local/_outputs/deep-llms_th2')
old=root/'6y3o-dot-pilot-20261020-a01'
pred=root/'6y3o-dot-R-20261021-a01'
burn_gate(pred)
assert burn_progress((pred/'supervised/burn.log').read_text())
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
assert digest('resources/llm_pretrain_burn.py')==BURN_HASH
assert not (root/'6y3o-dot-diagnostics-20261021-a01').exists()
for arm in ('P1-A','P1-B','N-A'):
 run=(root/'6y3o-dot-N-20261020-a01' if arm=='N-A' else old)/'prepared/runs'/arm
 for step in ([4768] if arm=='N-A' else [1000,2500,4768]):
  cp=run/f'checkpoint-{step}'
  for name in ('config.json','data_recipe.json','training_recipe.json','trainer_state.json'):
   assert (cp/name).stat().st_size>0
  assert len(list(cp.glob('*.safetensors')))>0
  assert json.loads((cp/'trainer_state.json').read_text())['global_step']==step
 print('CHECKPOINTS_OK',arm,flush=True)
 print('FINAL_EVAL',arm,(run/'final_eval.json').read_text(),flush=True)
receipt=json.loads((old/'prepared/runs/P1-A/training_recipe.json').read_text())
for name,h in receipt['code_sha256'].items():assert digest(name)==h,('Changed source',name)
args=json.loads((old/'prepared/P1-A.train.json').read_text())
for k in ('eval_data_dir','tokenizer_name','validation_manifest'):assert Path(args[k]).exists()
print('EVAL_ASSETS',json.dumps({k:args.get(k) for k in ('eval_data_dir','tokenizer_name','validation_manifest','data_cache_dir')}),flush=True)
print('GPU_LIVE',json.dumps(inspect()),flush=True)
print('FREE_DISK_GIB',shutil.disk_usage(root).free/1024**3,flush=True)
print('DIAGNOSTICS_PREFLIGHT_OK',flush=True)
PY
