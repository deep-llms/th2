#1 +60+a
#th2-tjx3-sdpa-repeat-200-preflight-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - <<'CHECK'
import json,subprocess,importlib.metadata as md
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
print('GPU_STATUS',json.dumps(inspect()),flush=True)
root=Path('/mnt/local/_outputs/deep-llms_th2/document-stability-200-20261004-a01/benchmark')
for name in ['data','eval','data.json','eval-data.json','correctness.json','sdpa_isolated/rank-0.json','fa4_isolated/rank-0.json']:
    path=root/name
    print('OLD_ARTIFACT',name,'exists',path.exists(),flush=True)
    if path.is_file():
        obj=json.loads(path.read_text())
        if name.endswith('rank-0.json'):obj={k:v for k,v in obj.items() if k not in ('log_history','steps')}
        print('OLD_CONTENT',name,json.dumps(obj),flush=True)
print('VERSIONS',json.dumps({p:md.version(p) for p in ['torch','transformers','accelerate','datasets','flash-attn-4','nvidia-cutlass-dsl']}),flush=True)
print('LOCAL_GIT',Path('.git').exists(),flush=True)
p=subprocess.run(['git','cat-file','-t','e06d5f9'],capture_output=True,text=True)
print('HISTORICAL_COMMIT_LOCAL',p.returncode,p.stdout.strip(),p.stderr.strip(),flush=True)
CHECK
