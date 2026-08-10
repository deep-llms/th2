#1
#th2-ppl-bytoken
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate eval
sleep 3

nvidia-smi | head -12

O=/opt/dlami/nvme/sparse_emb_outputs
CUDA_VISIBLE_DEVICES=0 python eval/ppl_bytoken.py \
    --checkpoint $O/original_ant/checkpoint-10000 \
    --eval-dir /opt/dlami/nvme/sparse_emb_data/Qwen_Qwen3-0.6B/eval \
    --tokenizer-name Qwen/Qwen3-0.6B --bf16 \
    > $O/original_ant/checkpoint-10000/ppl_bytoken.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval/ppl_bytoken.py \
    --checkpoint $O/v2_attn/checkpoint-10000 \
    --eval-dir /opt/dlami/nvme/sparse_emb_data/Qwen_Qwen3-0.6B/eval \
    --tokenizer-name Qwen/Qwen3-0.6B --bf16 \
    > $O/v2_attn/checkpoint-10000/ppl_bytoken.log 2>&1 &
wait

echo '=== original_ant summary ==='
cat $O/original_ant/checkpoint-10000/eval_ppl_bytoken_summary.json
echo '=== v2_attn summary ==='
cat $O/v2_attn/checkpoint-10000/eval_ppl_bytoken_summary.json
echo '=== logs tail ==='
tail -5 $O/original_ant/checkpoint-10000/ppl_bytoken.log $O/v2_attn/checkpoint-10000/ppl_bytoken.log
echo TH2 BYTOKEN DONE
