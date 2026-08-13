#1 +120+a
#th2-progress-2
echo '=== experiments.log ==='
cat /opt/dlami/nvme/sparse_emb_outputs/logs/experiments.log 2>/dev/null
echo '=== last loss lines ==='
grep -o "{'loss'[^}]*}" /opt/dlami/nvme/sparse_emb_outputs/logs/lowrank.log 2>/dev/null | tail -3
echo '=== gpu ==='
nvidia-smi | grep -E "MiB /" | head -4
echo '=== latest checkpoint ==='
ls -d /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-* 2>/dev/null | sort -V | tail -3
