#1 +30+a
#th2-stop-only-swt-serial-sampler-20260908-a01
set -euo pipefail
date -u
/mnt/local/conda-py311/envs/swt/bin/python -u - <<'PY'
import os, signal, select, time
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
        fd = os.pidfd_open(int(proc.name))
        if (proc/'cmdline').read_bytes().split(b'\0') != argv:
            os.close(fd)
            raise RuntimeError('Process identity changed; no signal sent')
        matched.append((int(proc.name), fd))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
assert len(matched) <= 1, 'Unexpected multiple samplers; refusing signals'
for pid, fd in matched:
    print('STOPPING_VERIFIED_CPU_SAMPLER', pid, flush=True)
    signal.pidfd_send_signal(fd, signal.SIGTERM)
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    if not poll.poll(20000):
        signal.pidfd_send_signal(fd, signal.SIGKILL)
        assert poll.poll(10000), 'Sampler did not exit'
    os.close(fd)
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
