#1 120+a
#debug-cuda-services
echo "=== Fabric Manager status ==="
systemctl status nvidia-fabricmanager 2>/dev/null || echo "fabricmanager service not found"
sleep 1

echo ""
echo "=== Persistence Daemon status ==="
systemctl status nvidia-persistenced 2>/dev/null || echo "persistenced service not found"
sleep 1

echo ""
echo "=== Try starting services (may need sudo) ==="
sudo systemctl start nvidia-fabricmanager 2>/dev/null && echo "fabricmanager started" || echo "cannot start fabricmanager (no sudo?)"
sudo systemctl start nvidia-persistenced 2>/dev/null && echo "persistenced started" || echo "cannot start persistenced (no sudo?)"
sleep 5

echo ""
echo "=== CUDA check after services ==="
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
