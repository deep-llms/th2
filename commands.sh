#1 +120+a
#th2-tail-train-loss-1
echo '=== log dir ==='
ls -la /opt/dlami/nvme/sparse_emb_outputs/logs/
echo '=== gpu ==='
nvidia-smi | head -20
echo '=== first loss lines (expect ~12.1 at step 10) ==='
for f in /opt/dlami/nvme/sparse_emb_outputs/logs/*.log; do
    echo "--- $f ---"
    grep -m 5 "'loss'" "$f"
    echo "--- last lines ---"
    tail -5 "$f"
done
