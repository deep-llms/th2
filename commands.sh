#1 +30+a
#th2-gpu-free-check
echo "=== UTC time ==="
date -u
echo "=== Host ==="
hostname
echo "=== NVIDIA status ==="
nvidia-smi
echo "=== Active GPU compute processes ==="
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
echo "=== Training-related host processes ==="
pgrep -af 'accelerate|torchrun|run_experiments|train_compositional|train_original_ant|finetune' || true
