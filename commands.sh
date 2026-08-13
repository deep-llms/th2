#1 +120+a
#th2-check-running
echo '=== GPU processes ==='
nvidia-smi | grep -E "MiB /|No running|python" | head -12
echo '=== training/eval processes ==='
pgrep -af "python|accelerate|eval_parallel|run_experiments" | grep -v pgrep | grep -v networkd | grep -v unattended | head -10 || echo "no processes"
echo '=== lowrank eval results exist? ==='
ls /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-10000/eval_ppl.json 2>/dev/null && echo "EVAL DONE" || echo "NO EVAL YET"
ls /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-1000/eval_ppl.json 2>/dev/null && echo "at least ckpt-1000 eval done" || echo "no evals at all"
echo TH2 CHECK DONE
