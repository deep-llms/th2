#1 +120+a
#th2-what-running
echo '=== ALL python processes ==='
ps aux | grep python | grep -v grep | head -15
echo '=== ALL GPU processes ==='
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
echo '=== finetune JSON list ==='
ls /opt/dlami/nvme/sparse_emb_outputs/finetune/*.json 2>/dev/null
echo '=== finetune log errors ==='
grep -l "Error\|Traceback\|FAILED" /opt/dlami/nvme/sparse_emb_outputs/finetune/*.log 2>/dev/null | head -5 || echo "no errors in logs"
echo '=== run_all run log tail ==='
f=$(ls -t _run_log_/*finetune*.log 2>/dev/null | head -1)
[ -n "$f" ] && tail -20 "$f" || echo "no finetune run log"
