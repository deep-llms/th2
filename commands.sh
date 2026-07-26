#1
#train-ant-experiments
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

nvidia-smi
sleep 3

# Clean previous outputs for a fresh start (keeps dataset cache)
rm -rf /opt/dlami/nvme/sparse_emb_outputs/original_ant
rm -rf /opt/dlami/nvme/sparse_emb_outputs/ant_ours
rm -rf /opt/dlami/nvme/sparse_emb_outputs/v2_attn
rm -rf /opt/dlami/nvme/sparse_emb_outputs/logs
sleep 3


rm -rf ~/.cache/huggingface/datasets
echo "HF cache removed"
sleep 3


# Remove cache/tmp files in sampled data
find /opt/dlami/nvme/sparse_emb_data -name "cache-*" -delete 2>/dev/null
find /opt/dlami/nvme/sparse_emb_data -name "tmp*" -delete 2>/dev/null
echo "Data cache/tmp files removed"
sleep 3




# Copy accelerate config
mkdir -p ~/.cache/huggingface/accelerate
cp resources/accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml
sleep 3

export WANDB_MODE=offline
python run_experiments.py --stop-at-step 10000 --log-dir /opt/dlami/nvme/sparse_emb_outputs/logs
