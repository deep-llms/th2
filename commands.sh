#1 +120+a
#th2-cuda-diag
echo '=== fabric manager status ==='
sudo systemctl status nvidia-fabricmanager 2>&1 | head -15
echo '=== nvidia-smi full ==='
nvidia-smi
echo '=== CUDA test (sparse_emb env) ==='
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate sparse_emb
python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
print('device 0:', torch.cuda.get_device_name(0))
x = torch.randn(100, 100, device='cuda')
print('matmul ok:', (x @ x).shape)
" 2>&1
echo '=== CUDA test (eval env) ==='
conda activate eval
python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
x = torch.randn(100, 100, device='cuda')
print('matmul ok:', (x @ x).shape)
" 2>&1
echo TH2 CUDA DIAG DONE
