#1 +120+a
#th2-force-rm-finetune
rm -rf /opt/dlami/nvme/sparse_emb_outputs/finetune
ls /opt/dlami/nvme/sparse_emb_outputs/finetune 2>/dev/null && echo "STILL EXISTS" || echo "GONE"
nvidia-smi | grep -E "MiB /|No running" | head -2
echo TH2 FORCE RM DONE
