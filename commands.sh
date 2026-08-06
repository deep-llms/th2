#1
#test-eval-benchmark
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate eval
sleep 3

CUDA_VISIBLE_DEVICES=0 python eval/eval_checkpoint.py \
    --checkpoint /opt/dlami/nvme/sparse_emb_outputs/v2_attn/checkpoint-10000 \
    --tokenizer-name Qwen/Qwen3-0.6B \
    --bf16 \
    --bench-only \
    --tasks hellaswag
