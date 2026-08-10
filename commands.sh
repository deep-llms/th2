#1 +120+a
#th2-eval-results-r2
echo '=== eval_parallel tail ==='
tail -8 eval_parallel.log
echo '=== eval procs ==='
pgrep -af "eval_parallel|eval_checkpoint" || echo "no eval processes"
for m in original_ant v2_attn; do
  for i in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000; do
    for kind in eval_ppl eval_benchmarks; do
      f=/opt/dlami/nvme/sparse_emb_outputs/$m/checkpoint-$i/$kind.json
      echo "===JSON $m $i $kind==="
      cat "$f" 2>/dev/null || echo MISSING
      echo
    done
  done
done
