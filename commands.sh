#1 +120+a
#th2-check-and-clean
echo '=== GPUs ==='
nvidia-smi | grep -E "MiB /|No running"
echo '=== output dirs ==='
ls -la /opt/dlami/nvme/sparse_emb_outputs/
echo '=== lowrank exists? ==='
ls -la /opt/dlami/nvme/sparse_emb_outputs/lowrank/ 2>/dev/null || echo "NOT EXIST"
echo '=== residual_ant exists? ==='
ls -la /opt/dlami/nvme/sparse_emb_outputs/residual_ant/ 2>/dev/null || echo "NOT EXIST"
echo '=== remove caches ==='
rm -rf ~/.cache/huggingface/datasets
find /opt/dlami/nvme/sparse_emb_data -name "cache-*" -delete 2>/dev/null
find /opt/dlami/nvme/sparse_emb_data -name "tmp*" -delete 2>/dev/null
echo "caches removed"
