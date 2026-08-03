#1
#clean-all
rm -rf /opt/dlami/nvme/sparse_emb_outputs/v2_attn
rm -rf /opt/dlami/nvme/sparse_emb_outputs/baseline
rm -rf /opt/dlami/nvme/sparse_emb_outputs/logs

rm -rf ~/.cache/huggingface/datasets
echo "HF cache removed"

find /opt/dlami/nvme/sparse_emb_data -name "cache-*" -delete 2>/dev/null
find /opt/dlami/nvme/sparse_emb_data -name "tmp*" -delete 2>/dev/null
echo "Data cache/tmp files removed"

echo "Done"
