#1
#th2-train-orig-and-v2
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

nvidia-smi
sleep 3

# Verify CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA FAILED'; print(f'CUDA OK: {torch.cuda.device_count()} GPUs')"
sleep 5

# Verify output dirs not exist (fresh start)
if [ -d /opt/dlami/nvme/sparse_emb_outputs/original_ant ]; then echo "ERROR: original_ant dir already exists"; exit 1; fi
if [ -d /opt/dlami/nvme/sparse_emb_outputs/v2_attn ]; then echo "ERROR: v2_attn dir already exists"; exit 1; fi

# Config
mkdir -p ~/.cache/huggingface/accelerate
cp resources/accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml

# Train original_ant (lam 1e-6, paper value) then v2_attn.
# Full-data schedule (~34.5k steps for LR/lambda), stopped manually at 10k.
export WANDB_MODE=offline
python run_experiments.py --experiments 0 2 --stop-at-step 10000 --log-dir /opt/dlami/nvme/sparse_emb_outputs/logs
