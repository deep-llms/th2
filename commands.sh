#1 +120+a
#th2-gpu-detail
echo '=== all 8 GPUs ==='
nvidia-smi | grep -E "MiB /|python" | head -20
echo '=== finetune processes ==='
pgrep -af "train.py" | grep -v pgrep | head -10
echo '=== per-GPU process count ==='
nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | head -10
echo '=== result count ==='
ls /opt/dlami/nvme/sparse_emb_outputs/finetune/*.json 2>/dev/null | wc -l
echo '=== latest logs ==='
ls -t /opt/dlami/nvme/sparse_emb_outputs/finetune/*.log 2>/dev/null | head -8 | while read f; do
    running=$(tail -1 "$f" | grep -c "epoch")
    name=$(basename "$f" .log)
    echo "  $name: $(tail -1 "$f" | head -c 80)"
done
