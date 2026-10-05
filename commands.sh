#1 +60+a
#th2-tjx3-p4-four-head-preflight-20261006-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
date -u
hostname
git rev-parse HEAD
TASK_PYTHON=/mnt/local/conda-py311/envs/train_env/bin/python3.11
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" -u - <<'PYREMOTE'
import json, importlib.metadata as md, shutil
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import approved_launcher,APPROVED_BURNS,BURN_HASH,GUARD_HASH,digest
from accelerate.commands.config.config_args import default_yaml_config_file
status=inspect()
print('INSPECTION',json.dumps(status),flush=True)
assert status['host']=='thiennh-p6-tjx3-worker-0'
assert len(status['gpus'])==8 and all('B200' in g['name'] for g in status['gpus'])
assert not status['guard_disabled']
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
approved={**APPROVED_BURNS,str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}
assert all(approved_launcher(p,approved) or approved_launcher(process(p)['ppid'],approved) for p in status['workers'])
print('PACKAGES',json.dumps({p:md.version(p) for p in ('torch','transformers','accelerate','datasets','safetensors')}))
recipe=json.loads(Path('proxy_heads.b200.json').read_text())
for key in ('config_name','tokenizer_name','data_dir','eval_data_dir'):assert Path(recipe[key]).exists(),key
baseline=Path('/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-2500-20261005-a01/supervised/run/baseline/seed-42/A')
config=json.loads((baseline/'train_config.json').read_text());result=json.loads((baseline/'result.json').read_text())
print('BASELINE',json.dumps(dict(path=str(baseline),step=result['global_step'],loss=result['evaluation']['eval_lm_loss'],train_fingerprint=config['train_fingerprint'],eval_fingerprint=config['eval_fingerprint'],pilot=config['pilot'])))
print('CAPACITY',json.dumps(dict(free_gib=shutil.disk_usage(baseline).free/2**30,accelerate_cache=default_yaml_config_file)))
print('READ_ONLY_PREFLIGHT_OK',flush=True)
PYREMOTE
