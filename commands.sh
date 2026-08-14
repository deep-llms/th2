#1
#th2-finetune-battery
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

nvidia-smi | head -12

python finetune/run_all.py \
    --checkpoints \
        lowrank=/opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-10000 \
        original_ant=/opt/dlami/nvme/sparse_emb_outputs/original_ant/checkpoint-10000 \
        v2_attn=/opt/dlami/nvme/sparse_emb_outputs/v2_attn/checkpoint-10000 \
    --tasks ag_news sst2 xnli paws_x hellaswag \
    --modes full probe \
    --seeds 42 123 456 \
    --output-dir /opt/dlami/nvme/sparse_emb_outputs/finetune \
    --device cuda
