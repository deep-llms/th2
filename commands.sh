#1 +60+a
#th2-verify-sparse-emb-fresh-b200-20260828-a01
set -euo pipefail

TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
test -x "$TASK_PYTHON"

echo '=== sparse_emb environment ==='
"$TASK_PYTHON" --version
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$TASK_PYTHON" - <<'PY'
import importlib.metadata
import sys

import accelerate
import datasets
import entmax
import huggingface_hub
import openai
import pyarrow
import scipy
import torch
import transformers
import wandb

assert sys.version_info[:2] == (3, 11), sys.version
expected = {
    "accelerate": "1.13.0",
    "datasets": "4.8.5",
    "transformers": "5.9.0",
}
for package, wanted in expected.items():
    actual = importlib.metadata.version(package)
    assert actual == wanted, (package, actual, wanted)

assert torch.cuda.is_available(), "torch.cuda is unavailable"
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
names = [torch.cuda.get_device_name(index) for index in range(8)]
assert all("B200" in name for name in names), names
for index in range(8):
    value = torch.ones(1, device=f"cuda:{index}")
    assert value.item() == 1.0
torch.cuda.synchronize()

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print(f"datasets={datasets.__version__}")
print(f"accelerate={accelerate.__version__}")
print(f"gpu_count={len(names)} gpu_names={names}")
print("imports=OK")
PY

echo 'TH2 SPARSE EMB ENV VERIFY OK'
