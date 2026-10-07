#1 +30+a
#th2-tjx3-downstream-inventory-20261007-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
date -u
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'REMOTE'
import json,os,subprocess
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
print('INSPECTION',json.dumps(inspect()),flush=True)
for name in ('eval','attention_bench'):
 p=Path('/mnt/local/conda-py311/envs')/name/'bin/python3.11'
 code="""import importlib.metadata as md,json,os
from pathlib import Path
result={}
for p in ('torch','transformers','datasets','accelerate','lm_eval','flash-attn-4','nvidia-cutlass-dsl'):
 try:result[p]=md.version(p)
 except md.PackageNotFoundError:result[p]=None
import datasets
from huggingface_hub.constants import HF_HUB_CACHE
result['datasets_cache']=str(datasets.config.HF_DATASETS_CACHE)
result['hub_cache']=str(HF_HUB_CACHE)
for key in ('datasets_cache','hub_cache'):
 p=Path(result[key]);result[key+'_children']=sorted(x.name for x in p.iterdir()) if p.exists() else []
print(json.dumps(result))
"""
 print('ENV',name,flush=True)
 if p.exists():subprocess.run([str(p),'-c',code],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'})
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-remaining-2500-20261006-a01/supervised/run/training/seed-42')
paths=[root/a for a in ('P7-simple','P7','P4-iso','P6','P5','P7-mlp','P7-kq','P7-ems','P4','P6-iso')]
paths.insert(0,Path('/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-fa4-2500-20261005-a01/supervised/run/baseline/seed-42/A'))
for p in paths:
 c=p/'checkpoint-2500';r=json.loads((p/'train_config.json').read_text())
 print('CHECKPOINT',str(c),json.dumps({'files':sorted(x.name for x in c.iterdir()),'model':r['model'],'step':json.loads((c/'trainer_state.json').read_text())['global_step']}),flush=True)
p=Path('/mnt/local/_data/deep-llms_th2');print('DATA_DIRS',sorted(x.name for x in p.iterdir()))
print('INVENTORY_DONE',flush=True)
REMOTE
