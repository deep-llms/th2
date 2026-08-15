#1 +120+a
#th2-ft-progress
nvidia-smi | grep -E "MiB /|python" | head -10
echo "=== jsons ==="
ls /opt/dlami/nvme/sparse_emb_outputs/finetune/*.json 2>/dev/null | wc -l
ls /opt/dlami/nvme/sparse_emb_outputs/finetune/*.json 2>/dev/null | head -5
echo "=== latest log ==="
f=$(ls -t /opt/dlami/nvme/sparse_emb_outputs/finetune/*.log 2>/dev/null | head -1)
[ -n "$f" ] && tail -5 "$f" || echo "no logs"
