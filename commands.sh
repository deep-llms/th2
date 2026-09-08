#1 +30+a
#th2-swt-qwen-readonly-preflight-20260908-a01
set -euo pipefail
date -u
hostname
pwd
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
df -h /mnt/local
/mnt/local/conda-py311/envs/swt/bin/python -u - <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
import torch, transformers, datasets, accelerate
print('VERSIONS', sys.executable, torch.__version__, transformers.__version__, datasets.__version__, accelerate.__version__)
raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
seen = set()
for line in raw.splitlines():
    pid = int(line.strip())
    for depth in range(6):
        if pid <= 1 or pid in seen:
            break
        seen.add(pid)
        proc = Path(f'/proc/{pid}')
        try:
            stat = (proc/'stat').read_text().rsplit(')',1)[1].split()
            argv = (proc/'cmdline').read_bytes().split(b'\0')
            print('GPU_ANCESTOR', json.dumps(dict(pid=pid, ppid=int(stat[1]), start=stat[19], argv=[v.decode(errors='replace') for v in argv if v])))
            pid = int(stat[1])
        except FileNotFoundError:
            break
for root in ['/mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B', '/mnt/local/_models/deep-llms_th2', '/mnt/local/.cache/huggingface/accelerate']:
    path = Path(root)
    print('PATH', root, 'EXISTS', path.exists())
    if path.is_dir():
        for p in sorted(path.iterdir()):
            print('ENTRY', str(p), 'directory' if p.is_dir() else p.stat().st_size)
data = Path('/mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B')
for p in sorted(data.glob('*.json')):
    if p.stat().st_size < 200000:
        print('DATA_METADATA', str(p), p.read_text())
for split in ['train/en', 'eval/en']:
    path = data/split
    print('SAMPLED_SPLIT', str(path), path.is_dir())
    if path.is_dir():
        shards = sorted(p for p in path.iterdir() if p.is_dir() and p.name.startswith('shard_'))
        total = 0
        for p in shards or [path]:
            ds = datasets.load_from_disk(str(p))
            assert 'text' in ds.column_names and len(ds) > 0
            assert all(isinstance(ds[i]['text'], str) and ds[i]['text'] for i in [0,len(ds)-1])
            total += len(ds)
        print('SAMPLED_READABLE', split, 'shards', len(shards), 'documents', total)
for p in [Path('/tmp/llm_pretrain_burn.py'), Path('/mnt/local/.cache/huggingface/accelerate/default_config.yaml')]:
    if p.is_file():
        print('FILE_HASH', str(p), hashlib.sha256(p.read_bytes()).hexdigest())
        print('FILE_CONTENT', str(p), p.read_text()[:20000])
print('SWT_QWEN_PREFLIGHT_COMPLETE')
PY
date -u
