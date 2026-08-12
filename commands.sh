#1
#th2-train-lowrank-resant
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

nvidia-smi
sleep 3

# Verify GPUs free
python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA OK: {torch.cuda.device_count()} GPUs')"

# Verify output dirs not exist (fresh start)
if [ -d /opt/dlami/nvme/sparse_emb_outputs/lowrank ]; then echo "ERROR: lowrank dir already exists"; exit 1; fi
if [ -d /opt/dlami/nvme/sparse_emb_outputs/residual_ant ]; then echo "ERROR: residual_ant dir already exists"; exit 1; fi

# Config
mkdir -p ~/.cache/huggingface/accelerate
cp resources/accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml

# Train lowrank (ALBERT, experiment 4) then residual_ant (experiment 5), stop at 10k
export WANDB_MODE=offline
python run_experiments.py --experiments 4 5 --stop-at-step 10000 --log-dir /opt/dlami/nvme/sparse_emb_outputs/logs
