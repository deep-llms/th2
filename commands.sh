#1 +60+a
#th2-tjx3-baseline-A-fa4-preflight-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/train_env/bin/python -u - <<'PY'
import json,subprocess,shutil
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import approved_launcher,APPROVED_BURNS,BURN_HASH,GUARD_HASH,digest
old=Path('/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-2500-20261005-a01')
state=inspect();print('GPU_INSPECTION',json.dumps(state),flush=True)
assert state['host']=='thiennh-p6-tjx3-worker-0' and not state['guard_disabled']
assert len(state['gpus'])==8 and all('B200' in gpu['name'] for gpu in state['gpus'])
approved={**APPROVED_BURNS,str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
assert all(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved) for pid in state['workers'])
assert json.loads((old/'supervised/supervisor.json').read_text())['training_status']=='ok'
prior=json.loads((old/'recipe.json').read_text());next_recipe=json.loads(Path('baseline_a_fa4.b200.json').read_text())
differences={k:[prior.get(k),next_recipe.get(k)] for k in prior.keys()|next_recipe.keys() if prior.get(k)!=next_recipe.get(k)}
print('RECIPE_DIFFERENCES',json.dumps(differences),flush=True)
assert set(differences)=={'attention_backend','output_dir'}
config=json.loads((old/'supervised/run/baseline/seed-42/A/train_config.json').read_text())
print('BASELINE_DATA_IDENTITY',json.dumps({k:config[k] for k in ['train_fingerprint','eval_fingerprint','tokens_per_update','world_size']}),flush=True)
print('DISK_FREE_GIB',shutil.disk_usage('/mnt/local').free/2**30,flush=True)
code='''import importlib.metadata as md,json,sys,torch
from accelerate.commands.config.config_args import default_yaml_config_file
selected={d.metadata['Name']:d.version for d in md.distributions() if d.metadata['Name'].lower().startswith(('nvidia-','torch','triton','flash-attn','nvidia-cutlass')) or d.metadata['Name'].lower() in ('transformers','accelerate','datasets','safetensors','numpy','pyarrow','huggingface-hub','tokenizers','wandb')}
print(json.dumps(dict(python=sys.executable,versions=selected,torch_cuda=torch.version.cuda,accelerate_cache=default_yaml_config_file)))
'''
for env in ['train_env','attention_bench']:
    python=Path('/mnt/local/conda-py311/envs')/env/'bin/python'
    assert python.is_file(),str(python)
    print('ENVIRONMENT',env,flush=True)
    subprocess.run([str(python),'-c',code],check=True)
    subprocess.run([str(python),'-m','pip','check'],check=True)
python='/mnt/local/conda-py311/envs/attention_bench/bin/python'
subprocess.run([python,'-c','from deep_kv.fa4 import load_kernel; print("FA4_IMPORT",load_kernel()[1])'],check=True)
print('FA4_READ_ONLY_PREFLIGHT_PASSED',flush=True)
PY
