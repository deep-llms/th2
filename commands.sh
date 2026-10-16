#1 +30+a
#th2-q359-ce7454-inspect-20261016-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-q359-worker-0
df -h /mnt/local
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
/mnt/local/conda-py311/envs/eval_fa4/bin/python -m scripts.inspect_burn
/mnt/local/conda-py311/envs/eval_fa4/bin/python - <<'PY'
import importlib, importlib.metadata, json
from pathlib import Path
for name in ('torch', 'torchvision', 'numpy', 'PIL', 'cv2', 'scipy'):
    try:
        m = importlib.import_module(name)
        print('IMPORT', name, getattr(m, '__version__', 'unknown'))
    except Exception as e:
        print('IMPORT_FAILED', name, type(e).__name__, str(e))
print('ENVS', [p.name for p in Path('/mnt/local/conda-py311/envs').iterdir()])
print('CE7454_DATA', [str(p) for p in Path('/mnt/local/_data/deep-llms_th2').glob('*ce7454*')])
print('CE7454_INSPECTION_COMPLETE')
PY
