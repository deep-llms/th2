#1 +120+a
#th2-final-clean-check
echo '=== GPU ==='
nvidia-smi | grep -E "MiB /|No running" | head -4
echo '=== finetune dir ==='
ls /opt/dlami/nvme/sparse_emb_outputs/finetune 2>/dev/null && echo "EXISTS" || echo "GONE"
echo '=== HF cache ==='
du -sh ~/.cache/huggingface/datasets 2>/dev/null || echo "NO CACHE"
echo '=== data cache ==='
find /opt/dlami/nvme/sparse_emb_data -name "cache-*" 2>/dev/null | wc -l
echo '=== processes ==='
pgrep -af "python" | grep -v pgrep | grep -v networkd | grep -v unattended | head -3 || echo "none"
echo '=== outputs ==='
ls -d /opt/dlami/nvme/sparse_emb_outputs/*/
echo TH2 FINAL CHECK
