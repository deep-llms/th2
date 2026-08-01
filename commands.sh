#1
#clean-before-train
rm -rf /opt/dlami/nvme/sparse_emb_outputs/original_ant
rm -rf /opt/dlami/nvme/sparse_emb_outputs/logs
rm -rf ~/.cache/huggingface/datasets
find /opt/dlami/nvme/sparse_emb_data -name "cache-*" -delete 2>/dev/null
find /opt/dlami/nvme/sparse_emb_data -name "tmp*" -delete 2>/dev/null
echo "Cleaned"
