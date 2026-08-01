#1
#verify-env
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

echo "=== System ==="
nvidia-smi | head -5
echo ""

echo "=== Fabric Manager ==="
systemctl status nvidia-fabricmanager --no-pager 2>&1 | grep "Active:"
echo ""

echo "=== Python + PyTorch ==="
python -c "
import torch
print('torch:', torch.__version__)
print('CUDA compiled:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPUs:', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1024**3:.0f} GB)')
    x = torch.randn(1000, 1000, device='cuda:0')
    y = x @ x.T
    print(f'matmul on GPU: OK ({y.shape})')
print()
print('bf16 support:', torch.cuda.is_bf16_supported())
"
echo ""

echo "=== Key packages ==="
python -c "
import transformers, datasets, accelerate, entmax
print('transformers:', transformers.__version__)
print('datasets:', datasets.__version__)
print('accelerate:', accelerate.__version__)
print('entmax: OK')
"
echo ""

echo "=== Accelerate config ==="
cat ~/.cache/huggingface/accelerate/default_config.yaml 2>/dev/null || echo 'not found'
echo ""

echo "=== Data check ==="
ls /opt/dlami/nvme/sparse_emb_data/Qwen_Qwen3-0.6B/train/ 2>/dev/null | head -10 || echo 'data dir not found'
