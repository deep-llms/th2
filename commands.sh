#1 +120+a
#th2-clean-new-exps
echo '=== remove lowrank + residual_ant output ==='
rm -rf /opt/dlami/nvme/sparse_emb_outputs/lowrank
rm -rf /opt/dlami/nvme/sparse_emb_outputs/residual_ant
echo '=== remove caches ==='
rm -rf ~/.cache/huggingface/datasets
find /opt/dlami/nvme/sparse_emb_data -name "cache-*" -delete 2>/dev/null
find /opt/dlami/nvme/sparse_emb_data -name "tmp*" -delete 2>/dev/null
echo '=== verify ==='
ls -la /opt/dlami/nvme/sparse_emb_outputs/
ls /opt/dlami/nvme/sparse_emb_outputs/lowrank 2>/dev/null || echo "lowrank: gone"
ls /opt/dlami/nvme/sparse_emb_outputs/residual_ant 2>/dev/null || echo "residual_ant: gone"
nvidia-smi | grep -E "MiB /|No running"
echo TH2 CLEAN
