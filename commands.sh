#1
#eval-v2-attn
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

python eval/eval_checkpoint.py \
    --checkpoint /opt/dlami/nvme/sparse_emb_outputs/v2_attn/checkpoint-10000 \
    --eval-dir /opt/dlami/nvme/sparse_emb_data/Qwen_Qwen3-0.6B/eval \
    --tokenizer-name Qwen/Qwen3-0.6B \
    --bf16 \
    --ppl-only
