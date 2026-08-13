#1 +120+a
#th2-check-done
echo '=== experiments.log ==='
cat /opt/dlami/nvme/sparse_emb_outputs/logs/experiments.log 2>/dev/null
echo '=== last loss ==='
grep -o "{'loss'[^}]*}" /opt/dlami/nvme/sparse_emb_outputs/logs/lowrank.log 2>/dev/null | tail -3
echo '=== latest checkpoints ==='
ls -d /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-* 2>/dev/null | sort -V | tail -3
echo '=== gpu ==='
nvidia-smi | grep -E "MiB /|No running" | head -4
echo '=== processes ==='
pgrep -af "run_experiments|train_compositional|accelerate" | grep -v pgrep | head -3 || echo "no training processes"
