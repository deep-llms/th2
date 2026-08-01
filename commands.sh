#1 +600+a
#fix-fabricmanager
sudo apt-get install -y --allow-downgrades nvidia-fabricmanager=595.71.05-1ubuntu1
sleep 3

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
    print('OK')
"
