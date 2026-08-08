#1
#retrain-h100-1-orig-and-v2
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

nvidia-smi
sleep 3

# Move old 16x-gradient runs aside, clean smoke leftovers
mv -v /opt/dlami/nvme/sparse_emb_outputs/original_ant /opt/dlami/nvme/sparse_emb_outputs/original_ant_old16x
mv -v /opt/dlami/nvme/sparse_emb_outputs/v2_attn /opt/dlami/nvme/sparse_emb_outputs/v2_attn_old16x
rm -rf /opt/dlami/nvme/sparse_emb_outputs/smoke_orig

# Guard: fresh output dirs (else Trainer would resume from old checkpoints)
if [ -d /opt/dlami/nvme/sparse_emb_outputs/original_ant ]; then echo "ERROR: original_ant dir still exists"; exit 1; fi
if [ -d /opt/dlami/nvme/sparse_emb_outputs/v2_attn ]; then echo "ERROR: v2_attn dir still exists"; exit 1; fi

# Config
mkdir -p ~/.cache/huggingface/accelerate
cp resources/accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml

# Train original_ant (lam 1e-6, paper value) then v2_attn, stop each at 10k
export WANDB_MODE=offline
python run_experiments.py --experiments 0 2 --stop-at-step 10000 --log-dir /opt/dlami/nvme/sparse_emb_outputs/logs
