#1 +120+a
#th2-fix-cuda
echo '=== downgrade fabric manager ==='
sudo apt-get install --allow-downgrades -y nvidia-fabricmanager=595.71.05-1ubuntu1 2>&1 | tail -5
echo '=== hold version ==='
sudo apt-mark hold nvidia-fabricmanager
echo '=== restart fabric manager ==='
sudo systemctl restart nvidia-fabricmanager
sleep 5
sudo systemctl status nvidia-fabricmanager 2>&1 | head -8
echo '=== disable unattended upgrades ==='
sudo systemctl disable unattended-upgrades 2>&1
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
echo '=== eval env CUDA test ==='
conda activate eval
python -c "
import torch
print('cuda available:', torch.cuda.is_available())
x = torch.randn(100, 100, device='cuda')
print('matmul ok:', (x @ x).shape)
"
echo TH2 CUDA FIXED
