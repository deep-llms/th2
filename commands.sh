#1 +120+a
#th2-ft-status
nvidia-smi | grep -E "MiB /|No running" | head -4
echo "=== jsons ==="
ls /opt/dlami/nvme/sparse_emb_outputs/finetune/*.json 2>/dev/null | wc -l
echo "=== summary ==="
ls /opt/dlami/nvme/sparse_emb_outputs/finetune/summary.md 2>/dev/null && echo EXISTS || echo NO
echo "=== procs ==="
pgrep -af "train.py\|run_all" | grep -v pgrep | head -3 || echo "none"
