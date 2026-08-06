#1
#eval-h100-1
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate eval
sleep 3

pip install entmax
sleep 3

for exp in original_ant v2_attn; do
    for step in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000; do
        echo "========== ${exp} checkpoint-${step} =========="
        CUDA_VISIBLE_DEVICES=0 python eval/eval_checkpoint.py \
            --checkpoint /opt/dlami/nvme/sparse_emb_outputs/${exp}/checkpoint-${step} \
            --eval-dir /opt/dlami/nvme/sparse_emb_data/Qwen_Qwen3-0.6B/eval \
            --tokenizer-name Qwen/Qwen3-0.6B \
            --bf16
    done
done
