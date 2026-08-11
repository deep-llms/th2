#1
#th2-anchor-usage-2
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate eval
sleep 3

O=/opt/dlami/nvme/sparse_emb_outputs
echo '===== original_ant (static, exact over vocab, freq-weighted) ====='
python scripts/analyze_anchor_usage.py \
    --checkpoint $O/original_ant/checkpoint-10000 \
    --freq-npz resources/token_freq_sample10.npz \
    --output-prefix $O/original_ant/checkpoint-10000/anchor_usage
echo '===== v2_attn (context, empirical over eval text) ====='
python scripts/analyze_anchor_usage.py \
    --checkpoint $O/v2_attn/checkpoint-10000 \
    --eval-dir /opt/dlami/nvme/sparse_emb_data/Qwen_Qwen3-0.6B/eval \
    --tokenizer-name Qwen/Qwen3-0.6B --max-tokens-per-lang 500000 \
    --output-prefix $O/v2_attn/checkpoint-10000/anchor_usage
echo TH2 ANCHOR DONE
