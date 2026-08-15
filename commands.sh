#1 +120+a
#th2-ft-check2
ls /opt/dlami/nvme/sparse_emb_outputs/finetune/*.json 2>/dev/null | wc -l
ls /opt/dlami/nvme/sparse_emb_outputs/finetune/summary.md 2>/dev/null && echo "DONE" || echo "RUNNING"
nvidia-smi | grep -c "python"
