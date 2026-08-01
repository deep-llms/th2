#1
#verify-cuda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

systemctl is-active nvidia-fabricmanager && echo "Fabric manager: OK" || echo "Fabric manager: DOWN"
sleep 3

python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPUs:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU 0:', torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device='cuda:0')
    print('Tensor on GPU: OK')
else:
    print('CUDA FAILED')
"
