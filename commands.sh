#1 +30+a
#th2-ccm-readonly-preflight-20260913-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-8mgy-worker-0
pwd
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
/mnt/local/conda-py311/envs/train_env/bin/python -B - <<'PY'
import importlib, json, subprocess
from pathlib import Path
for name in ('torch','transformers','numpy','pyarrow','pytest'):
    module=importlib.import_module(name)
    print('ENV',name,getattr(module,'__version__','unknown'),flush=True)
import torch
print('CUDA',torch.version.cuda,torch.cuda.is_available(),torch.cuda.device_count(),flush=True)
pids={int(x.strip()) for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines() if x.strip()}
for pid in sorted(pids):
    seen=set()
    while pid>1 and pid not in seen:
        seen.add(pid)
        proc=Path('/proc')/str(pid)
        try:
            stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
            argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            print('GPU_ANCESTRY',json.dumps(dict(pid=pid,ppid=int(stat[1]),start_ticks=stat[19],argv=argv)),flush=True)
            pid=int(stat[1])
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            print('PROCESS_CHANGED',pid,flush=True); break
for path in ('/mnt/local/_data/deep-llms_th2/data/raw/en','/mnt/local/_models/deep-llms_th2',
             '/mnt/local/_outputs/deep-llms_th2','/mnt/local/_gpu_guard'):
    p=Path(path)
    print('DIRECTORY',path,'exists',p.exists(),flush=True)
    if p.is_dir(): print('CHILDREN',json.dumps(sorted(x.name for x in p.iterdir())[:100]),flush=True)
root=Path('/mnt/local/_data/deep-llms_th2/data/raw/en')
files=sorted(root.glob('*.parquet'))
print('EN_FILES',len(files),'EN_BYTES',sum(p.stat().st_size for p in files),flush=True)
PY
test -s /tmp/llm_pretrain_burn.py
sha256sum /tmp/llm_pretrain_burn.py
sed -n '1,220p' /tmp/llm_pretrain_burn.py
command -v tmux
df -h /mnt/local
free -h
date -u
echo CCM_READONLY_PREFLIGHT_COMPLETE
