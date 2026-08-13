#1 +120+a
#th2-running-ok
echo '=== gpu ==='
nvidia-smi | grep -E "MiB /" | head -4
echo '=== eval process ==='
pgrep -af "eval_parallel\|eval_checkpoint" | grep -v pgrep | head -3 || echo "no eval processes"
echo '=== any results yet? ==='
ls /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-1000/eval_ppl.json 2>/dev/null && echo "ckpt-1000 ppl DONE" || echo "ckpt-1000 not done yet"
