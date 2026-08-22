#1 +60+a
#th2-probe-pytorch-cuda-for-gpu-burn-20260822
set -euo pipefail

echo '=== th2 PyTorch CUDA burn probe ==='
date -u
hostname

echo '=== preflight ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
TASK_EXISTING_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"
test -z "$TASK_EXISTING_GPU_PIDS"

CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 - <<'PY'
import torch

print('torch_version', torch.__version__)
print('torch_cuda_version', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
print('visible_gpu_count', torch.cuda.device_count())
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
print('device_name', torch.cuda.get_device_name(0))

x = torch.randn((8192, 8192), device='cuda', dtype=torch.float16)
y = torch.randn((8192, 8192), device='cuda', dtype=torch.float16)
z = x @ y
torch.cuda.synchronize()
print('matmul_ok', tuple(z.shape), z.dtype)
PY

echo '=== post-probe: GPU must be free ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
TASK_REMAINING_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"
test -z "$TASK_REMAINING_GPU_PIDS"

echo 'TH2 PYTORCH CUDA BURN PROBE OK'
