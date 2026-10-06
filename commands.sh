#1 +30+a
#th2-tjx3-unrun-inventory-20261006-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PY'
import json, shutil, importlib.metadata as md
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect, process
from scripts.train_then_burn import approved_launcher, APPROVED_BURNS, BURN_HASH, GUARD_HASH, digest
root=Path('/mnt/local/_outputs/deep-llms_th2')
state=inspect()
approved={**APPROVED_BURNS,str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}
state['approved_burn_workers']={str(pid):bool(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved)) for pid in state['workers']}
print('FRESH_GPU_INSPECTION',json.dumps(state),flush=True)
print('GUARD_HASH',digest('/mnt/local/_gpu_guard/gpu_guard.sh'),flush=True)
print('DISK',json.dumps(shutil.disk_usage(root)._asdict()),flush=True)
print('ENVIRONMENT',json.dumps({p:md.version(p) for p in ('torch','transformers','accelerate','flash-attn-4','datasets','nvidia-cutlass-dsl')}),flush=True)
for path in sorted(root.rglob('result.json')):
 try:
  result=json.loads(path.read_text());config_path=path.parent/'train_config.json'
  cfg=json.loads(config_path.read_text()) if config_path.is_file() else {}
  if result.get('arm') in ('A','P4-iso','P4','P5','P6','P6-iso','P4-iso-4h','P4-4h','P7','P7-kq','P7-mlp','P7-ems','P7-simple'):
   print('RESULT',json.dumps(dict(path=str(path),arm=result.get('arm'),step=result.get('global_step'),status=result.get('status'),backend=cfg.get('pilot',{}).get('attention_backend','sdpa'),seed=cfg.get('training',{}).get('seed'))),flush=True)
 except (OSError,ValueError) as e: print('UNREADABLE_RESULT',str(path),type(e).__name__,flush=True)
previous=root/'p7-checks-20261006-a01/supervised'
for name in ('supervisor.json','burn-verified.json'):
 print('LAST_HANDOFF',name,(previous/name).read_text(),flush=True)
print('BURN_PROGRESS',(previous/'burn.log').read_text()[-2000:],flush=True)
recipe=json.loads(Path('proxy_heads.b200.json').read_text())
for key in ('config_name','tokenizer_name','data_dir','eval_data_dir'):
 p=Path(recipe[key]);print('INPUT',key,str(p),p.exists(),flush=True)
print('INVENTORY_FINISHED',flush=True)
PY
