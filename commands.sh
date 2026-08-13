#1 +120+a
#th2-pull-train-progress
echo '=== experiments.log ==='
cat /opt/dlami/nvme/sparse_emb_outputs/logs/experiments.log 2>/dev/null || echo "no experiments.log"
echo '=== gpu ==='
nvidia-smi | grep -E "MiB /" | head -4
echo '=== last loss lines ==='
for f in /opt/dlami/nvme/sparse_emb_outputs/logs/*.log; do
    echo "--- $f ---"
    grep -o "{'loss'[^}]*}" "$f" | tail -3
done
