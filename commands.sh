#1 +120+a
#th2-swt-stop-and-clean-a03-only-20260908-a01
set -euo pipefail
date -u
test "$(hostname)" = thiennh-p6-oish-worker-0
test "$PWD" = /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt
test "$CONDA_DEFAULT_ENV" = swt
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
python - <<'PY'
import hashlib
from pathlib import Path
assert hashlib.sha256(Path('scripts/stop_capacity_queue.py').read_bytes()).hexdigest() == 'd87e5396b728b4e6439a6992c078124c96142c20f1fa5c75b2bf3ee7b68f5397'
for name in (
    '/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_10k_s42_20260908_a03',
    '/mnt/local/_data/deep-llms_th2/swt/qwen_en_map160_batch1000',
):
    p = Path(name)
    assert p.is_dir() and not p.is_symlink() and p.resolve(strict=True) == p, p
print('STOP_HELPER_AND_CLEANUP_PATHS_VERIFIED', flush=True)
PY
python -u -m scripts.stop_capacity_queue --queue-pid 152404 --queue-start 163046836 \
  --run-root /mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_10k_s42_20260908_a03 --stop
python - <<'PY'
import os
from pathlib import Path
import shutil
from scripts.gpu_status import require_free
from scripts.reclaim_verified_burn import identity

targets = [
    Path('/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_10k_s42_20260908_a03'),
    Path('/mnt/local/_data/deep-llms_th2/swt/qwen_en_map160_batch1000'),
]
protected = [
    Path('/mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B'),
    Path('/mnt/local/_models/deep-llms_th2/Qwen3-0.6B'),
    Path('/mnt/local/.cache/huggingface/datasets'),
]
require_free(list(range(8)))
active_names = {'train.py','train_capacity_b200.sh','scripts.prepare_capacity_cache',
                'scripts.benchmark_capacity_batch','benchmark_batches_b200.sh'}
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) <= 1 or int(proc.name) == os.getpid():
        continue
    try:
        record = identity(int(proc.name))
    except (FileNotFoundError, ProcessLookupError):
        continue
    assert not any(Path(arg).name in active_names for arg in record['argv']), record
for path in targets:
    assert path.is_dir() and not path.is_symlink() and path.resolve(strict=True) == path, path
    assert not os.path.ismount(path)
def inventory(root):
    assert root.is_dir(), root
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in root.rglob('*') if p.is_file())
before = {str(p): inventory(p) for p in protected}
assert all(before.values())
for path in targets:
    print('DELETE_AUTHORIZED_EXACT_TARGET', path, flush=True)
    shutil.rmtree(path)
    assert not os.path.lexists(path)
    print('VERIFIED_ABSENT_INCLUDING_ALL_CACHE_AND_TMP_FILES', path, flush=True)
for path in protected:
    assert inventory(path) == before[str(path)], path
    print('PRESERVED_UNCHANGED', path, flush=True)
for name in ('/mnt/local/_outputs/deep-llms_th2/swt','/mnt/local/_data/deep-llms_th2/swt'):
    leftovers = [str(p) for p in Path(name).rglob('*') if p.name.startswith(('cache-','tmp-'))]
    print('STAGEWISE_CACHE_TMP_LEFTOVERS', name, len(leftovers), flush=True)
    assert not leftovers, leftovers[:10]
require_free(list(range(8)))
print('SWT_A03_CANCELLED_AND_CLEANED_ALL_EIGHT_GPUS_FREE_NO_RESTART', flush=True)
PY
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
