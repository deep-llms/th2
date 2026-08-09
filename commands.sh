#1 +120+a
#th2-check-progress-3
echo '=== experiments.log ==='
cat /opt/dlami/nvme/sparse_emb_outputs/logs/experiments.log
echo '=== v2_attn last loss lines ==='
grep -o "{'loss'[^}]*}" /opt/dlami/nvme/sparse_emb_outputs/logs/v2_attn.log | tail -3
echo '=== v2_attn last checkpoints ==='
ls -d /opt/dlami/nvme/sparse_emb_outputs/v2_attn/checkpoint-* 2>/dev/null | sort -V | tail -3
echo '=== running procs ==='
pgrep -af "run_experiments|accelerate|train_" | head -5 || echo "no training processes"
echo '=== gpu ==='
nvidia-smi | grep -E "MiB|%" | head -8
