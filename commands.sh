#1 +120+a
#th2-rerun-lowrank-tied
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

echo '=== th2 low-rank tied rerun preflight ==='
date -u
hostname
echo "conda=$CONDA_DEFAULT_ENV"
python --version
nvidia-smi

TASK_GPU_PROCESSES="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"
if [ -n "$TASK_GPU_PROCESSES" ]; then
    echo 'ERROR: GPU compute processes are active; refusing to launch.'
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
    exit 1
fi

TASK_OUTPUT_DIR=/opt/dlami/nvme/sparse_emb_outputs/lowrank_tied
if [ -e "$TASK_OUTPUT_DIR" ]; then
    echo "ERROR: $TASK_OUTPUT_DIR already exists; refusing to resume or overwrite."
    exit 1
fi

echo '=== launching low-rank tied rerun ==='
bash scripts/train_lowrank_tied.sh
