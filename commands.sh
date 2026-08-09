#1 +120+a
#th2-check-progress-2
echo '=== log dir ==='
ls -la /opt/dlami/nvme/sparse_emb_outputs/logs/ /opt/dlami/nvme/sparse_emb_outputs/
echo '=== experiments.log ==='
cat /opt/dlami/nvme/sparse_emb_outputs/logs/experiments.log
echo '=== latest loss lines per log ==='
for f in /opt/dlami/nvme/sparse_emb_outputs/logs/*.log; do
    echo "--- $f ---"
    grep -o "{'loss'[^}]*}" "$f" | tail -3
done
echo '=== gpu ==='
nvidia-smi | head -12
