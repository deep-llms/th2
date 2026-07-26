#1 +120+a
#setup-and-clean
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

pip install entmax
sleep 3

python -c "from entmax import entmax15; print('entmax OK')"
sleep 3

# Clean previous outputs
rm -rf /opt/dlami/nvme/sparse_emb_outputs/original_ant
rm -rf /opt/dlami/nvme/sparse_emb_outputs/ant_ours
rm -rf /opt/dlami/nvme/sparse_emb_outputs/v2_attn
rm -rf /opt/dlami/nvme/sparse_emb_outputs/logs

# Clean HF dataset cache
rm -rf ~/.cache/huggingface/datasets
echo "HF cache removed"

# Clean cache/tmp files in data dir
find /opt/dlami/nvme/sparse_emb_data -name "cache-*" -delete 2>/dev/null
find /opt/dlami/nvme/sparse_emb_data -name "tmp*" -delete 2>/dev/null
echo "Data cache/tmp files removed"

# Copy accelerate config
mkdir -p ~/.cache/huggingface/accelerate
cp resources/accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml

echo "Done"
