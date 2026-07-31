#1
#debug-cuda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

echo "=== nvidia-smi ==="
nvidia-smi
sleep 3

echo "=== NVIDIA driver version ==="
cat /proc/driver/nvidia/version 2>/dev/null || echo "not found"

echo "=== CUDA toolkit in torch ==="
python -c "import torch; print('torch:', torch.__version__); print('CUDA compiled:', torch.version.cuda); print('cuDNN:', torch.backends.cudnn.version())"
sleep 3

echo "=== CUDA available (after nvidia-smi) ==="
python -c "
import torch
print('cuda.is_available():', torch.cuda.is_available())
print('device_count():', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device_name(0):', torch.cuda.get_device_name(0))
    x = torch.randn(2, 2, device='cuda:0')
    print('tensor on GPU:', x.device, x.dtype)
    print('GPU OK')
else:
    print('CUDA NOT AVAILABLE')
    print('Trying to force init...')
    try:
        torch.cuda.init()
        print('After init: available=', torch.cuda.is_available())
    except Exception as e:
        print('init error:', e)
"

echo "=== entmax ==="
python -c "from entmax import entmax15; print('entmax OK')"

echo "=== Done ==="
