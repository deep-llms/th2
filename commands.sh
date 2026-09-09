#1 +60+a
#th2-swt-stop-verified-burn-and-eval-preflight-20260909-a02
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-oish-worker-0
cd /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt
test "$(command -v python)" = /mnt/local/conda-py311/envs/swt/bin/python
python - <<'PY'
import scripts.reclaim_verified_burn as helper
from pathlib import Path
assert Path(helper.__file__).resolve() == Path.cwd()/'scripts/reclaim_verified_burn.py'
print('BURN_HELPER_VERIFIED', helper.__file__)
PY
sha256sum /tmp/llm_pretrain_burn.py
python -m scripts.reclaim_verified_burn --burn-path /tmp/llm_pretrain_burn.py --gpus 0 1 2 3 4 5 6 7
python -m scripts.reclaim_verified_burn --burn-path /tmp/llm_pretrain_burn.py --gpus 0 1 2 3 4 5 6 7 --stop
sleep 30
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
echo SWT_BURNS_STOPPED_ALL_GPUS_FREE
for TASK_ENV in swt swt_eval lm_eval; do
    TASK_PYTHON="/mnt/local/conda-py311/envs/$TASK_ENV/bin/python"
    if [ -x "$TASK_PYTHON" ]; then
        "$TASK_PYTHON" - <<'PY'
import sys, importlib.metadata as md
print('ENVIRONMENT', sys.executable)
for name in ('torch', 'transformers', 'datasets', 'accelerate', 'lm_eval'):
    try: print(name, md.version(name))
    except md.PackageNotFoundError: print(name, 'MISSING')
PY
    fi
done
python3 - <<'PY'
from pathlib import Path
root=Path('/mnt/local/_data/deep-llms_th2/benchmarks')
print('BENCHMARK_ROOT_EXISTS',root.is_dir())
if root.is_dir():
    for p in sorted(root.glob('*/*')):
        if p.is_dir(): print('SNAPSHOT',p,'FILES',sum(1 for f in p.rglob('*') if f.is_file()))
print('SWT_EVAL_PREFLIGHT_COMPLETE')
PY
