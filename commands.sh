#1 +60+a
#th2-tjx3-proxy-fa4-screen-preflight-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - <<'PY'
import json,copy,hashlib,shutil,importlib.metadata as md
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import approved_launcher,APPROVED_BURNS,BURN_HASH
from deep_kv.report import comparable_config
from transformers import AutoTokenizer,TrainingArguments
from deep_kv.packing import preprocess_dataset
from train import load_text
project=Path.cwd();root=Path('/mnt/local/_outputs/deep-llms_th2')
base=root/'proxy-baseline-A-fa4-2500-20261005-a01/supervised/run/baseline/seed-42/A'
smoke=root/'proxy-fa4-smoke-20261005-a01/supervised/run/seed-42/A'
status=inspect();print('GPU_STATUS',json.dumps(status),flush=True)
assert status['host']=='thiennh-p6-tjx3-worker-0' and not status['guard_disabled']
approved={**APPROVED_BURNS,str(project/'resources/llm_pretrain_burn.py'):BURN_HASH}
assert all(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved) for pid in status['workers'])
versions={name:md.version(name) for name in ('torch','transformers','accelerate','datasets','flash-attn-4','nvidia-cutlass-dsl')}
print('VERSIONS',json.dumps(versions),flush=True)
assert versions==dict(torch='2.14.1',transformers='5.9.0',accelerate='1.13.0',datasets='4.8.5',**{'flash-attn-4':'4.0.0b33','nvidia-cutlass-dsl':'4.8.0'})
assert shutil.disk_usage(root).free>180*2**30
read=lambda p:json.loads(p.read_text())
recipe=read(project/'baseline_a_fa4.b200.json')
old=read(base/'train_config.json');new=read(smoke/'train_config.json')
assert old['pilot']['attention_backend']=='fa4' and old['world_size']==8
result=read(base/'result.json');assert result['global_step']==2500 and result['status']=='stopped'
assert result['attention_runtime']['version']=='4.0.0b33'
checkpoint=base/'checkpoint-2500'
for name in ['model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json',*[f'rng_state_{i}.pth' for i in range(8)]]:
    assert (checkpoint/name).stat().st_size>0,name
assert read(checkpoint/'trainer_state.json')['global_step']==2500
assert read(root/'proxy-fa4-smoke-20261005-a01/supervised/run/smoke-validation.json')['status']=='passed'
new['training']['logging_steps']=recipe['logging_steps']
for key in ('eval_rows','monitor_rows'):new['data'][key]=recipe[key]
# Verify the full validation prefix using the normal cached train.py pipeline.
tokenizer=AutoTokenizer.from_pretrained(recipe['tokenizer_name'],local_files_only=True)
if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
args=TrainingArguments(output_dir='/tmp/proxy-fa4-read-only-preflight',use_cpu=True,report_to=[])
validation=preprocess_dataset(load_text(recipe['eval_data_dir']),tokenizer,recipe['block_size'],args,
    num_proc=1,isolate_documents=True).select(range(recipe['eval_rows']))
assert validation._fingerprint==old['eval_fingerprint'],(validation._fingerprint,old['eval_fingerprint'])
new['eval_fingerprint']=validation._fingerprint
assert comparable_config(old)==comparable_config(new),'Baseline differs from current smoke beyond smoke-only settings'
print('PREFLIGHT_PASSED',json.dumps(dict(baseline=str(base),step=result['global_step'],
    baseline_config_sha256=hashlib.sha256((base/'train_config.json').read_bytes()).hexdigest(),
    train_fingerprint=old['train_fingerprint'],eval_fingerprint=validation._fingerprint,
    full_current_recipe_matches=True,free_gib=shutil.disk_usage(root).free/2**30)),flush=True)
print('BASELINE_CONFIG',json.dumps(old),flush=True)
PY
