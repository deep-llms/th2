#1 +30+a
#th2-6y3o-stock-fa2-offline-install-20261020-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline HF_HUB_DISABLE_TELEMETRY=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
TASK_PYTHON=/mnt/local/conda-py311/envs/train_fa2_t28/bin/python3.11
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" - <<'INSTALL'
from pathlib import Path
import hashlib,importlib.metadata,json,subprocess,sys
import torch
assert Path(sys.prefix)==Path('/mnt/local/conda-py311/envs/train_fa2_t28')
assert torch.__version__.split('+')[0]=='2.8.0' and torch.version.cuda=='12.8'
assert 'sm_100' in torch._C._cuda_getArchFlags() and torch._C._GLIBCXX_USE_CXX11_ABI
try: importlib.metadata.version('flash-attn')
except importlib.metadata.PackageNotFoundError: pass
else: raise RuntimeError('Refusing to replace an existing flash-attn installation')
wheel=Path('/mnt/local/_data/deep-llms_th2/build_inputs/flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl')
assert wheel.stat().st_size==256238631
assert hashlib.sha256(wheel.read_bytes()).hexdigest()=='92aabf18ecfcf5d620ac7fe97fafe88f83e714b7a790e4bf219dff4a404cfb38'
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',str(wheel)],check=True)
print('STOCK_FA2_OFFLINE_INSTALL_OK',flush=True)
INSTALL
python3 scripts/attention_env_inventory.py --host thiennh-p6-6y3o-worker-0 --compare /mnt/local/_outputs/deep-llms_th2/6y3o-attention-study-20261020-a01/inventory-before.json --output /mnt/local/_outputs/deep-llms_th2/6y3o-attention-compare-20261020-a01/preflight.json
