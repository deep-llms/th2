#1 +120+a
#th2-verify-eval-loading
echo '=== eval_parallel progress ==='
tail -20 eval_parallel.log 2>/dev/null || ls -la
echo '=== per-checkpoint eval logs: loading lines ==='
for f in /opt/dlami/nvme/sparse_emb_outputs/original_ant/checkpoint-1000/eval.log /opt/dlami/nvme/sparse_emb_outputs/v2_attn/checkpoint-1000/eval.log; do
    echo "--- $f ---"
    grep -iE "compositional|inferred|train_config|arm|error" "$f" | head -8
    tail -3 "$f"
done
echo '=== gpu ==='
nvidia-smi | grep -E "MiB /" | head -8
