#1 +120+a
#th2-post-wait-verify
echo '=== GPU ==='
nvidia-smi | grep -E "MiB /|No running" | head -4
echo '=== eval results ==='
for i in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000; do
    ppl=$([ -f /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-$i/eval_ppl.json ] && echo Y || echo N)
    bench=$([ -f /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-$i/eval_benchmarks.json ] && echo Y || echo N)
    echo "  ckpt-$i: ppl=$ppl bench=$bench"
done
echo '=== processes ==='
pgrep -af "eval_parallel\|eval_checkpoint\|python" | grep -v pgrep | grep -v networkd | grep -v unattended | head -3 || echo "none"
echo TH2 VERIFY DONE
