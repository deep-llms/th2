#1 +30+a
#th2-stop-only-swt-serial-sampler-20260908-a02
set -euo pipefail
date -u
/mnt/local/conda-py311/envs/swt/bin/python -u - <<'PY'
import os, signal, time
from pathlib import Path
target = b'/mnt/local/_data/deep-llms_th2/swt/english_gpt2_10b_seed0_20260907_a01'
matched = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) <= 1:
        continue
    try:
        argv = (proc/'cmdline').read_bytes().split(b'\0')
        if b'-m' not in argv or b'capacity_allocation.data' not in argv or b'--output' not in argv:
            continue
        if argv[argv.index(b'--output')+1] != target:
            continue
        start = (proc/'stat').read_text().rsplit(')', 1)[1].split()[19]
        matched.append((int(proc.name), start, argv))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
assert len(matched) <= 1, 'Unexpected multiple samplers; refusing signals'
def still_same(pid, start):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return stat[19] == start and stat[0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False
for pid, start, argv in matched:
    assert pid > 1 and pid != os.getpid()
    assert still_same(pid, start)
    assert Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') == argv
    print('STOPPING_VERIFIED_CPU_SAMPLER', pid, flush=True)
    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        if not still_same(pid, start):
            break
        time.sleep(1)
    assert not still_same(pid, start), 'Sampler still alive; no further signals sent'
    print('SAMPLER_EXIT_CONFIRMED', pid, flush=True)
if not matched:
    print('NO_MATCHING_SAMPLER_PROCESS')
root = Path(target.decode())
before = {p.name:p.stat().st_size for p in root.glob('*.bin')}
time.sleep(3)
after = {p.name:p.stat().st_size for p in root.glob('*.bin')}
assert before == after, 'Outputs still changing'
print('PARTIAL_OUTPUT_PRESERVED', after)
print('SWT_SERIAL_SAMPLER_STOPPED')
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
date -u
