#1 +120+a
#th2-cancel-clean-finetune
echo '=== kill finetune ==='
for i in 1 2 3; do
    pkill -f "finetune/run_all.py" 2>/dev/null
    pkill -f "finetune/train.py" 2>/dev/null
    sleep 5
done
nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read pid; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null && echo "killed $pid"
done
sleep 5
echo '=== remove finetune outputs ==='
rm -rf /opt/dlami/nvme/sparse_emb_outputs/finetune
echo '=== remove caches ==='
rm -rf ~/.cache/huggingface/datasets
echo '=== verify ==='
nvidia-smi | grep -E "MiB /|No running" | head -4
ls /opt/dlami/nvme/sparse_emb_outputs/finetune 2>/dev/null || echo "finetune dir: gone"
echo TH2 FINETUNE CLEANED
