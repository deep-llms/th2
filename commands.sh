#1 +120+a
#th2-full-check
echo '=== GPUs ==='
nvidia-smi | grep -E "MiB /|No running"
echo '=== processes ==='
pgrep -af "python|accelerate|train" | grep -v pgrep | head -5 || echo "no processes"
echo '=== output dirs ==='
ls -la /opt/dlami/nvme/sparse_emb_outputs/
echo '=== lowrank? ==='
ls -d /opt/dlami/nvme/sparse_emb_outputs/lowrank 2>/dev/null && echo "EXISTS" || echo "NOT EXIST"
echo '=== residual_ant? ==='
ls -d /opt/dlami/nvme/sparse_emb_outputs/residual_ant 2>/dev/null && echo "EXISTS" || echo "NOT EXIST"
echo '=== HF dataset cache? ==='
du -sh ~/.cache/huggingface/datasets 2>/dev/null || echo "NO CACHE"
echo '=== data dir caches? ==='
find /opt/dlami/nvme/sparse_emb_data -name "cache-*" 2>/dev/null | head -5 || echo "none"
echo TH2 CHECK DONE
