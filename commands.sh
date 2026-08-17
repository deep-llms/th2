#1 +120+a
#th2-finetune-lowrank-tied-10k
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate eval
sleep 3

echo '=== th2 low-rank tied generative finetune preflight ==='
date -u
hostname
echo "conda=$CONDA_DEFAULT_ENV"
python --version
nvidia-smi

TASK_CHECKPOINT=/opt/dlami/nvme/sparse_emb_outputs/lowrank_tied/checkpoint-10000
TASK_OUTPUT_DIR=/opt/dlami/nvme/sparse_emb_outputs/finetune_tied

TASK_GPU_PROCESSES="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"
if [ -n "$TASK_GPU_PROCESSES" ]; then
    echo 'ERROR: GPU compute processes are active; refusing to start finetuning.'
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
    exit 1
fi

python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8, torch.cuda.device_count(); print('CUDA OK: 8 GPUs')"

for TASK_FILE in config.json embedding.pt trainer_state.json eval_ppl.json eval_benchmarks.json; do
    if [ ! -f "$TASK_CHECKPOINT/$TASK_FILE" ]; then
        echo "ERROR: missing checkpoint/eval file: $TASK_CHECKPOINT/$TASK_FILE"
        exit 1
    fi
done

if [ -e "$TASK_OUTPUT_DIR" ]; then
    echo "ERROR: finetune output already exists; refusing to resume or overwrite: $TASK_OUTPUT_DIR"
    exit 1
fi
echo "fresh output path confirmed: $TASK_OUTPUT_DIR"

echo '=== starting 3 tasks x 3 seeds on 8 GPUs ==='
python finetune/run_all.py \
    --checkpoints lowrank_tied="$TASK_CHECKPOINT" \
    --tasks hellaswag arc_easy xnli \
    --seeds 42 123 456 \
    --num-gpus 8 \
    --output-dir "$TASK_OUTPUT_DIR"

test -f "$TASK_OUTPUT_DIR/summary.md"
echo 'TH2 LOWRANK TIED FINETUNE DONE'
