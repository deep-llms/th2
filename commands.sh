#1 +30+a
#th2-lowrank-tied-preflight
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

echo "=== UTC time ==="
date -u
echo "=== Conda environment ==="
echo "$CONDA_DEFAULT_ENV"
python --version
echo "=== Host ==="
hostname
echo "=== NVIDIA status ==="
nvidia-smi
echo "=== Active GPU compute processes ==="
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
echo "=== Training-related host processes ==="
pgrep -af 'accelerate|torchrun|run_experiments|train_compositional|train_original_ant|finetune' || true
