#1 +120+a
#th2-final-done-check
echo '=== GPU ==='
nvidia-smi | grep -E "MiB /|No running" | head -4
echo '=== ckpt-9000 bench missing? ==='
ls /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-9000/eval_benchmarks.json 2>/dev/null && echo "EXISTS" || echo "MISSING"
cat /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-9000/eval.log 2>/dev/null | tail -5
echo TH2 DONE CHECK
