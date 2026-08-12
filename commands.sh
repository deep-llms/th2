#1
#th2-train-lowrank
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

nvidia-smi | head -12
python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA OK: {torch.cuda.device_count()} GPUs')"

if [ -d /opt/dlami/nvme/sparse_emb_outputs/lowrank ]; then echo "ERROR: lowrank dir already exists"; exit 1; fi

mkdir -p ~/.cache/huggingface/accelerate
cp resources/accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml

# ALBERT-style low-rank control: full-data schedule, manual stop at 10k
export WANDB_MODE=offline
python run_experiments.py --experiments 4 --stop-at-step 10000 --log-dir /opt/dlami/nvme/sparse_emb_outputs/logs
