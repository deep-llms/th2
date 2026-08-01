#1
#test-cuda-fresh
echo "=== Fabric Manager status ==="
systemctl status nvidia-fabricmanager --no-pager 2>&1 | head -5
sleep 10

echo ""
echo "=== Fresh CUDA test (new process, 10s after FM) ==="
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

python -c "
import torch
print('torch:', torch.__version__)
print('CUDA compiled:', torch.version.cuda)
print('available:', torch.cuda.is_available())
print('count:', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    x = torch.randn(2,2,device='cuda:0')
    print('tensor on GPU: OK')
else:
    print('--- DEBUG ---')
    import os
    print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'))
    print('LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH', 'not set'))
    import subprocess
    r = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    print('nvidia-smi -L:', r.stdout.strip())
"
