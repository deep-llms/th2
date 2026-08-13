#1 +120+a
#th2-fix-cuda-retry
echo '=== restart fabric manager ==='
sudo systemctl restart nvidia-fabricmanager
sleep 15
sudo systemctl status nvidia-fabricmanager 2>&1 | head -5
echo '=== CUDA test ==='
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate sparse_emb
python -c "
import torch
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
print('device 0:', torch.cuda.get_device_name(0))
x = torch.randn(100, 100, device='cuda')
print('matmul ok:', (x @ x).shape)
"
echo TH2 FIX RETRY DONE
