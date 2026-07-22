#1
#cleanup-and-setup
# Kill any running training
pkill -f "train.py" 2>/dev/null
pkill -f "accelerate" 2>/dev/null
sleep 5

# Check GPUs
nvidia-smi

# Remove dataset cache
rm -rf ~/.cache/huggingface/datasets
echo "HF cache removed"

# Remove cache/tmp files in sampled data
find /opt/dlami/nvme/sparse_emb_data -name "cache-*" -delete 2>/dev/null
find /opt/dlami/nvme/sparse_emb_data -name "tmp*" -delete 2>/dev/null
echo "Data cache/tmp files removed"

# Copy accelerate config
mkdir -p ~/.cache/huggingface/accelerate
cp resources/accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml
echo "Accelerate config copied"

echo "done"
