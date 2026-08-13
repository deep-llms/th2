#1 +120+a
#th2-check-stale
echo '=== partial eval.log files ==='
for i in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000; do
    f=/opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-$i/eval.log
    [ -f "$f" ] && echo "  ckpt-$i: eval.log exists ($(wc -c < $f) bytes)" || echo "  ckpt-$i: clean"
done
echo '=== any eval_ppl/eval_benchmarks ==='
ls /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-*/eval_ppl.json /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-*/eval_benchmarks.json 2>/dev/null || echo "none"
echo TH2 STALE CHECK DONE
