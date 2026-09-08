#1 +30+a
#th2-readonly-swt-sampling-liveness-20260908-a01
set -euo pipefail
date -u
hostname
export CUDA_VISIBLE_DEVICES=""
/mnt/local/conda-py311/envs/swt/bin/python -u - <<'PY'
from pathlib import Path
import json, time
root = Path('/mnt/local/_data/deep-llms_th2/swt/english_gpt2_10b_seed0_20260907_a01')
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        argv = (proc / 'cmdline').read_bytes().split(b'\0')
        if b'capacity_allocation.data' in argv:
            print('SAMPLER_PROCESS', proc.name, (proc/'stat').read_text())
            print('SAMPLER_ARGV', [a.decode(errors='replace') for a in argv if a])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
def snapshot():
    return {p.name: {'bytes': p.stat().st_size, 'mtime': p.stat().st_mtime}
            for p in root.iterdir() if p.is_file()}
before = snapshot()
print('FILES_BEFORE', json.dumps(before), flush=True)
time.sleep(10)
after = snapshot()
print('FILES_AFTER', json.dumps(after), flush=True)
for split in ('train','validation','test'):
    name = split+'.bin'
    if name in before and name in after:
        print('TOKEN_FILE_GROWTH_BYTES', split, after[name]['bytes']-before[name]['bytes'])
manifest = root/'manifest.json'
print('COMPLETION_MANIFEST_EXISTS', manifest.is_file())
if manifest.is_file():
    print(manifest.read_text())
PY
df -h /mnt/local
date -u
