#1
#train-original-ant
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

# Clean previous outputs + cache
rm -rf /opt/dlami/nvme/sparse_emb_outputs/original_ant
rm -rf /opt/dlami/nvme/sparse_emb_outputs/logs
rm -rf ~/.cache/huggingface/datasets
find /opt/dlami/nvme/sparse_emb_data -name "cache-*" -delete 2>/dev/null
find /opt/dlami/nvme/sparse_emb_data -name "tmp*" -delete 2>/dev/null
echo "Cleaned"
sleep 10

# Verify CUDA
systemctl is-active nvidia-fabricmanager && echo "Fabric manager: OK" || echo "Fabric manager: DOWN (OK with NCCL_NVLS_ENABLE=0)"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA FAILED'; print(f'CUDA OK: {torch.cuda.device_count()} GPUs')"
sleep 5

# Copy accelerate config
mkdir -p ~/.cache/huggingface/accelerate
cp resources/accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml

# Train
export WANDB_MODE=offline
python run_experiments.py --experiments 0 --stop-at-step 10000 --log-dir /opt/dlami/nvme/sparse_emb_outputs/logs
