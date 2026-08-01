#1
#fix-fabricmanager
echo "=== Current versions ==="
echo "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "Fabric Manager installed: $(dpkg -l | grep fabricmanager | awk '{print $3}' || echo 'unknown')"

echo ""
echo "=== Installing matching fabric manager for driver 595.71.05 ==="
sudo apt-get update
sudo apt-get install -y nvidia-fabricmanager-595=595.71.05-1
sleep 3

echo ""
echo "=== Starting fabric manager ==="
sudo systemctl start nvidia-fabricmanager
sudo systemctl status nvidia-fabricmanager --no-pager
sleep 5

echo ""
echo "=== CUDA check ==="
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate sparse_emb
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    x = torch.randn(2,2,device='cuda:0')
    print('Tensor on GPU: OK')
"
