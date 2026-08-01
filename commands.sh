#1
#fix-fabricmanager
echo "=== Available fabric manager versions ==="
apt-cache madison nvidia-fabricmanager | head -5

echo ""
echo "=== Installing matching version ==="
sudo apt-get install -y nvidia-fabricmanager=595.71.05-1ubuntu1
sleep 3

echo ""
echo "=== Starting fabric manager ==="
sudo systemctl restart nvidia-fabricmanager
sleep 5
sudo systemctl status nvidia-fabricmanager --no-pager

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
