#1 +120+a
#th2-finetune-progress
echo '=== GPU ==='
nvidia-smi | grep -E "MiB /|python" | head -4
echo '=== finetune output dir ==='
ls /opt/dlami/nvme/sparse_emb_outputs/finetune/*.json 2>/dev/null | head -10 || echo "no results yet"
echo '=== run log tail ==='
tail -30 /opt/dlami/nvme/sparse_emb_outputs/finetune/run_all.log 2>/dev/null || echo "no run_all.log"
echo '=== latest run log from _run_log_ ==='
f=$(ls -t _run_log_/*finetune*.log 2>/dev/null | head -1)
[ -n "$f" ] && tail -30 "$f" || echo "no finetune run log"
echo TH2 PROGRESS
