#1 +120+a
#th2-ft-count
ls /opt/dlami/nvme/sparse_emb_outputs/finetune/*.json 2>/dev/null | wc -l
ls /opt/dlami/nvme/sparse_emb_outputs/finetune/summary.md 2>/dev/null && echo "SUMMARY EXISTS" || echo "NO SUMMARY"
nvidia-smi | grep -c "python"
