#1 +120+a
#th2-finetune-done
echo '=== GPU ==='
nvidia-smi | grep -E "MiB /|No running" | head -4
echo '=== processes ==='
pgrep -af "finetune\|train.py" | grep -v pgrep | head -3 || echo "none"
echo '=== result JSONs ==='
ls /opt/dlami/nvme/sparse_emb_outputs/finetune/*.json 2>/dev/null | wc -l
echo '=== any failed? ==='
for f in /opt/dlami/nvme/sparse_emb_outputs/finetune/*.log; do
    [ -f "$f" ] && grep -l "FAILED\|Error\|Traceback" "$f" 2>/dev/null
done || echo "no failures"
echo '=== summary exists? ==='
ls /opt/dlami/nvme/sparse_emb_outputs/finetune/summary.md 2>/dev/null && echo YES || echo NO
echo TH2 FT CHECK
