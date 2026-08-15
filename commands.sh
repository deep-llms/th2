#1 +120+a
#th2-verify-clean
nvidia-smi | grep -E "MiB /|No running" | head -4
ls /opt/dlami/nvme/sparse_emb_outputs/finetune 2>/dev/null && echo "EXISTS" || echo "GONE"
ls -d /opt/dlami/nvme/sparse_emb_outputs/*/ 2>/dev/null
echo TH2 OK
