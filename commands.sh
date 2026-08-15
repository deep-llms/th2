#1 +120+a
#th2-kill-clean-ft
for i in 1 2 3 4 5; do
    pkill -f "finetune" 2>/dev/null
    pkill -f "train.py" 2>/dev/null
    pkill -f "run_all" 2>/dev/null
    sleep 5
done
nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read pid; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null && echo "killed $pid"
done
sleep 5
rm -rf /opt/dlami/nvme/sparse_emb_outputs/finetune
rm -rf ~/.cache/huggingface/datasets
nvidia-smi | grep -E "MiB /|No running" | head -4
ls /opt/dlami/nvme/sparse_emb_outputs/finetune 2>/dev/null || echo "finetune: gone"
echo TH2 KILLED AND CLEANED
