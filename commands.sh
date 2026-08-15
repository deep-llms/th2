#1 +120+a
#th2-clean-finetune
echo '=== kill any finetune ==='
pkill -f "finetune" 2>/dev/null; sleep 5
nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read pid; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null && echo "killed $pid"
done
sleep 5
echo '=== remove finetune outputs ==='
rm -rf /opt/dlami/nvme/sparse_emb_outputs/finetune
echo '=== verify ==='
nvidia-smi | grep -E "MiB /|No running" | head -4
ls /opt/dlami/nvme/sparse_emb_outputs/finetune 2>/dev/null || echo "finetune: gone"
echo TH2 CLEAN
