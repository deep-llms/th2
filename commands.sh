#1 +30+a
#th2-sampling-prerequisites-20260911-a01
set -euo pipefail
date -u
hostname
free -h
nproc
df -h /mnt/local
/mnt/local/conda-py311/envs/train_env/bin/python -B - <<'PY'
import os
from pathlib import Path
print('CPU_AFFINITY',len(os.sched_getaffinity(0)))
model=Path('/mnt/local/_models/deep-llms_th2/Qwen3-0.6B')
for name in ('config.json','tokenizer.json','tokenizer_config.json'):
    p=model/name
    print('TOKENIZER_FILE',str(p),p.stat().st_size if p.exists() else 'ABSENT')
out=Path('/mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B')
print('SAMPLING_OUTPUT_EXISTS',out.exists())
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        argv=(proc/'cmdline').read_bytes().split(b'\0')
        if argv and b'python' in Path(os.fsdecode(argv[0])).name.encode() and any(x.endswith(b'prepare_data.py') for x in argv):
            print('EXISTING_SAMPLER',proc.name,[os.fsdecode(x) for x in argv if x])
    except (FileNotFoundError,ProcessLookupError,PermissionError): pass
PY
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
echo SAMPLING_PREREQUISITE_CHECK_COMPLETE
