#1 +30+a
#th2-swt-clean-stopped-run-and-owned-cache-20260908-a01
set -euo pipefail
date -u
hostname
/mnt/local/conda-py311/envs/swt/bin/python - <<'PY'
import os
from pathlib import Path
import shutil
import socket
from scripts.gpu_status import snapshot

targets = [
    Path('/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_10k_s42_20260908_a02'),
    Path('/mnt/local/_outputs/deep-llms_th2/swt/batch_benchmark_20260908_a01'),
    Path('/mnt/local/_data/deep-llms_th2/swt/qwen_en_map160_batch1000'),
]
protected = [
    Path('/mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B'),
    Path('/mnt/local/_models/deep-llms_th2/Qwen3-0.6B'),
    Path('/mnt/local/.cache/huggingface/datasets'),
]
assert socket.gethostname() == 'thiennh-p6-oish-worker-0'
gpu_rows = snapshot()
assert len(gpu_rows) == 8 and all(not row['pids'] for row in gpu_rows), gpu_rows
active_names = {'train.py', 'train_capacity_b200.sh', 'benchmark_batches_b200.sh',
                'scripts.benchmark_capacity_batch', 'scripts.prepare_capacity_cache'}
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        args = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
    except (FileNotFoundError, ProcessLookupError):
        continue
    assert not any(Path(arg).name in active_names for arg in args), (proc.name, args)
for path in targets:
    assert path.is_dir() and not path.is_symlink() and path.resolve(strict=True) == path, path
    assert not os.path.ismount(path), path
for path in protected:
    assert path.is_dir(), path
def inventory(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in root.rglob('*') if p.is_file())
before = {str(p): inventory(p) for p in protected}
assert all(before.values())
assert (targets[0] / 'B0' / 'checkpoint-6250').is_dir()
assert (targets[1] / 'benchmark_complete.json').is_file()
assert list(targets[2].glob('*-packed*.arrow'))
for path in targets:
    print('DELETE_AUTHORIZED_EXACT_TARGET', path, flush=True)
    shutil.rmtree(path)
    assert not os.path.lexists(path)
    print('VERIFIED_ABSENT', path, flush=True)
for path in protected:
    assert inventory(path) == before[str(path)], path
    print('PRESERVED_UNCHANGED', path, flush=True)
gpu_rows = snapshot()
assert len(gpu_rows) == 8 and all(not row['pids'] for row in gpu_rows), gpu_rows
print('ALL_EIGHT_GPUS_FREE', flush=True)
print('SWT_AUTHORIZED_CLEANUP_COMPLETE', flush=True)
PY
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
df -h /mnt/local
