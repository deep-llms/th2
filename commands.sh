#1
#train-original-ant
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

# Ensure fabric manager is running (required for H100 multi-GPU CUDA)
if ! systemctl is-active --quiet nvidia-fabricmanager; then
    echo "Fabric manager not running, starting..."
    sudo systemctl start nvidia-fabricmanager
    sleep 10
fi
systemctl is-active nvidia-fabricmanager && echo "Fabric manager: OK" || echo "WARN: fabric manager not running"

nvidia-smi
sleep 3

# Clean previous outputs
rm -rf /opt/dlami/nvme/sparse_emb_outputs/original_ant
rm -rf /opt/dlami/nvme/sparse_emb_outputs/logs

# Copy accelerate config
mkdir -p ~/.cache/huggingface/accelerate
cp resources/accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml
sleep 3

export WANDB_MODE=offline
python run_experiments.py --experiments 0 --stop-at-step 10000 --log-dir /opt/dlami/nvme/sparse_emb_outputs/logs
