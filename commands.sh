#1
#th2-gen-finetune
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate eval
sleep 3

echo '=== pre-flight checks ==='
nvidia-smi | head -12
python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA OK: {torch.cuda.device_count()} GPUs')"

if [ -d /opt/dlami/nvme/sparse_emb_outputs/finetune ]; then
    echo "ERROR: finetune dir already exists — not a fresh run"
    exit 1
fi
echo "finetune dir: clean (does not exist)"

echo '=== starting finetune ==='
python finetune/run_all.py \
    --checkpoints \
        lowrank=/opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-10000 \
        original_ant=/opt/dlami/nvme/sparse_emb_outputs/original_ant/checkpoint-10000 \
        v2_attn=/opt/dlami/nvme/sparse_emb_outputs/v2_attn/checkpoint-10000 \
    --tasks hellaswag arc_easy xnli \
    --seeds 42 123 456 \
    --output-dir /opt/dlami/nvme/sparse_emb_outputs/finetune
