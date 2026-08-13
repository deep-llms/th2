#1 +120+a
#th2-cancel-clean-all
echo '=== kill all ==='
for i in 1 2 3 4 5; do
    pkill -f run_experiments.py 2>/dev/null
    pkill -f train_compositional.py 2>/dev/null
    pkill -f "accelerate launch" 2>/dev/null
    sleep 5
done
nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read pid; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null && echo "killed $pid"
done
sleep 10
echo '=== remove lowrank + residual_ant ==='
rm -rf /opt/dlami/nvme/sparse_emb_outputs/lowrank
rm -rf /opt/dlami/nvme/sparse_emb_outputs/residual_ant
echo '=== remove caches ==='
rm -rf ~/.cache/huggingface/datasets
find /opt/dlami/nvme/sparse_emb_data -name "cache-*" -delete 2>/dev/null
find /opt/dlami/nvme/sparse_emb_data -name "tmp*" -delete 2>/dev/null
echo '=== verify ==='
nvidia-smi | grep -E "MiB /|No running"
ls -la /opt/dlami/nvme/sparse_emb_outputs/
ls /opt/dlami/nvme/sparse_emb_outputs/lowrank 2>/dev/null || echo "lowrank: gone"
ls /opt/dlami/nvme/sparse_emb_outputs/residual_ant 2>/dev/null || echo "residual_ant: gone"
find /opt/dlami/nvme/sparse_emb_data -name "cache-*" 2>/dev/null | wc -l
echo TH2 CLEAN
