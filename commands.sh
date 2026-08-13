#1 +120+a
#th2-pre-eval-check
echo '=== any partial eval results in lowrank checkpoints? ==='
for d in /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-*/; do
    ppl=$(ls "$d"eval_ppl.json 2>/dev/null && echo "HAS_PPL" || echo "no_ppl")
    bench=$(ls "$d"eval_benchmarks.json 2>/dev/null && echo "HAS_BENCH" || echo "no_bench")
    log=$(ls "$d"eval.log 2>/dev/null && echo "HAS_LOG" || echo "no_log")
    echo "$(basename $d): $ppl $bench $log"
done
echo '=== train_config.json exists? ==='
cat /opt/dlami/nvme/sparse_emb_outputs/lowrank/train_config.json 2>/dev/null | head -3 || echo "MISSING"
echo '=== embedding.pt in checkpoint-10000? ==='
ls -la /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-10000/embedding.pt 2>/dev/null || echo "MISSING"
echo '=== gpu free? ==='
nvidia-smi | grep -E "No running"
echo TH2 PRE-EVAL DONE
