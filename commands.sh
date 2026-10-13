#1 +15+a
#th2-q359-screen22-preflight-20261013-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PY'
import importlib.metadata as md,json,shutil,socket,sys
from datetime import datetime,timezone
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import approved_launcher,APPROVED_BURNS,BURN_HASH,GUARD_HASH,digest
print('AT',datetime.now(timezone.utc).isoformat(),'HOST',socket.gethostname(),flush=True)
print('PYTHON',sys.executable,flush=True)
print('VERSIONS',json.dumps({p:md.version(p) for p in ('torch','transformers','accelerate','datasets','flash-attn-4','nvidia-cutlass-dsl','wandb')}),flush=True)
record=inspect();print('INSPECTION',json.dumps(record),flush=True)
approved={**APPROVED_BURNS,str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}
print('BURN_IDENTITIES_APPROVED',all(approved_launcher(p,approved) or approved_launcher(process(p)['ppid'],approved) for p in record['workers']),flush=True)
print('GUARD_HASH_MATCH',digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH,flush=True)
recipe=json.loads(Path('proxy_heads.12layer.b200.json').read_text())
for key in ('config_name','tokenizer_name','data_dir','eval_data_dir'):
 p=Path(recipe[key]);print('ASSET',key,str(p),p.exists(),flush=True)
for name in ('/mnt/local/_outputs/deep-llms_th2','/mnt/local/_data/deep-llms_th2'):
 print('DISK',name,shutil.disk_usage(name)._asdict(),flush=True)
print('PREFLIGHT_INSPECTION_COMPLETE',flush=True)
PY
