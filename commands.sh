#1
#retrain-h100-1-orig-and-v2-clean
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

nvidia-smi
sleep 3

echo '=== state before cleanup ==='
ls -la /opt/dlami/nvme/sparse_emb_outputs/

# Old runs were ALREADY moved to *_old16x by the killed relaunch — do NOT mv again
# (mv onto an existing dir would nest inside it). Verify the archives exist:
if [ ! -d /opt/dlami/nvme/sparse_emb_outputs/original_ant_old16x ]; then echo "ERROR: original_ant_old16x archive missing"; exit 1; fi
if [ ! -d /opt/dlami/nvme/sparse_emb_outputs/v2_attn_old16x ]; then echo "ERROR: v2_attn_old16x archive missing"; exit 1; fi

# Remove PARTIAL dirs left by the killed retrain (else Trainer resumes from a partial checkpoint)
rm -rf /opt/dlami/nvme/sparse_emb_outputs/original_ant
rm -rf /opt/dlami/nvme/sparse_emb_outputs/v2_attn
rm -rf /opt/dlami/nvme/sparse_emb_outputs/smoke_orig

if [ -d /opt/dlami/nvme/sparse_emb_outputs/original_ant ]; then echo "ERROR: original_ant still exists"; exit 1; fi
if [ -d /opt/dlami/nvme/sparse_emb_outputs/v2_attn ]; then echo "ERROR: v2_attn still exists"; exit 1; fi

# Config
mkdir -p ~/.cache/huggingface/accelerate
cp resources/accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml

# Train original_ant (lam 1e-6, paper value) then v2_attn, stop each at 10k
export WANDB_MODE=offline
python run_experiments.py --experiments 0 2 --stop-at-step 10000 --log-dir /opt/dlami/nvme/sparse_emb_outputs/logs
