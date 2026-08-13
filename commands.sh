#1 +120+a
#th2-eval-done-check
echo '=== eval results exist? ==='
for i in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000; do
    ppl=$(ls /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-$i/eval_ppl.json 2>/dev/null && echo "Y" || echo "N")
    bench=$(ls /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-$i/eval_benchmarks.json 2>/dev/null && echo "Y" || echo "N")
    echo "  ckpt-$i: ppl=$ppl bench=$bench"
done
echo '=== processes ==='
pgrep -af "eval_parallel|eval_checkpoint|python" | grep -v pgrep | grep -v networkd | grep -v unattended | head -5 || echo "no processes"
echo '=== gpu ==='
nvidia-smi | grep -E "MiB /|No running" | head -4
