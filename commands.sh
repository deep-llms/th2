#1
#clean-test-eval
pkill -f eval_checkpoint.py 2>/dev/null
sleep 5

rm -f /opt/dlami/nvme/sparse_emb_outputs/v2_attn/checkpoint-10000/eval_benchmarks.json
rm -f /opt/dlami/nvme/sparse_emb_outputs/v2_attn/checkpoint-10000/eval_ppl.json
rm -f /opt/dlami/nvme/sparse_emb_outputs/v2_attn/checkpoint-10000/eval.log

echo "cleaned"
ls /opt/dlami/nvme/sparse_emb_outputs/v2_attn/checkpoint-10000/eval* 2>/dev/null || echo "no eval files left"
