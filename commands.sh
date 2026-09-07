#1 +60+a
#th2-swt-clone-verify-20260907-a01
set -euo pipefail
date -u
hostname
test "$PWD" = /mnt/local/deep-llms_th2
sha256sum -c <<'HASHES'
e9e83a70c29c4e65a9f7956e0b53cddec1d7a85dc4ac72bc36e087b514f199e1  capacity_allocation/modeling.py
37d0f631549a50cfa07d9ffd77170d20ec7af314f7e8644ab5c89abc2d37de85  capacity_allocation/data.py
27a5985eeb3e5ee6f7036a5736848db08a6ab72a9e2418cee7fb2529b2775ef1  train_capacity.py
HASHES
echo STAGEWISE_CODE_VERIFIED
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
df -h /mnt/local
test -x /mnt/local/conda-py311/envs/sparse_emb/bin/python
test ! -e /mnt/local/conda-py311/envs/swt
/mnt/local/conda-py311/bin/conda create --yes --offline --prefix /mnt/local/conda-py311/envs/swt --clone /mnt/local/conda-py311/envs/sparse_emb
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt
test "$CONDA_PREFIX" = /mnt/local/conda-py311/envs/swt
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -u - <<'PY'
import importlib.metadata as m, json, subprocess, sys
from pathlib import Path
names = ['torch', 'transformers', 'accelerate', 'datasets', 'numpy', 'pyarrow', 'tokenizers', 'safetensors']
original = json.loads(subprocess.check_output(['/mnt/local/conda-py311/envs/sparse_emb/bin/python', '-c', 'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in '+repr(names)+'}))'], text=True))
current = {n:m.version(n) for n in names}
print(json.dumps({'python': sys.executable, 'versions': current}, indent=2))
assert current == original, (current, original)
import torch, transformers, datasets, accelerate, pyarrow
print('SWT_IMPORTS_AND_SOURCE_VERSIONS_VERIFIED', flush=True)
raw = Path('/mnt/local/_data/deep-llms_th2/data/raw/en')
files = sorted(raw.glob('*.parquet'))
print(json.dumps({'raw_root':str(raw),'files':len(files),'bytes':sum(p.stat().st_size for p in files)}), flush=True)
for p in files:
    print(p.name, p.stat().st_size)
PY
python -m unittest discover -s tests -v
date -u
echo SWT_ENV_AND_CPU_TESTS_OK
