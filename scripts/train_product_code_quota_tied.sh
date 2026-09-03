#!/bin/bash
# Hashed v2: v1-preserving coverage-quota head with an exactly tied output.

set -euo pipefail

if [ "${CONDA_DEFAULT_ENV:-}" != "sparse_emb" ]; then
    if [ -x /mnt/local/conda-py311/bin/conda ]; then
        eval "$(/mnt/local/conda-py311/bin/conda shell.bash hook)"
    elif [ -x /mnt/local/conda/bin/conda ]; then
        eval "$(/mnt/local/conda/bin/conda shell.bash hook)"
    elif [ -x "$HOME/miniconda3/bin/conda" ]; then
        eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    else
        echo "ERROR: activate sparse_emb before launching"
        exit 1
    fi
    conda activate sparse_emb
fi

TASK_PYTHON="${SPARSE_EMB_PYTHON:-$(command -v python3.11)}"
TASK_MODEL_DIR="${SPARSE_EMB_MODEL_DIR:-Qwen/Qwen3-0.6B}"
TASK_DATA_DIR="${SPARSE_EMB_DATA_DIR:-/mnt/local/_data/sparse_embedding/data/Qwen_Qwen3-0.6B/train}"
TASK_OUTPUT_BASE="${SPARSE_EMB_OUTPUT_BASE:-/mnt/local/_outputs/sparse_embedding}"
TASK_IMPORTANCE_PATH=resources/token_importance_quota.npz
TASK_HEAD_SIZE="${PRODUCT_CODE_QUOTA_HEAD_SIZE:-6144}"

case "$TASK_HEAD_SIZE" in
    6144) TASK_NUM_BUCKETS=3072 ;;
    8192) TASK_NUM_BUCKETS=2560 ;;
    *)
        echo "ERROR: PRODUCT_CODE_QUOTA_HEAD_SIZE must be 6144 or 8192"
        exit 1
        ;;
esac

TASK_EXPERIMENT="product_code_quota_h${TASK_HEAD_SIZE}"
TASK_OUTPUT_DIR="$TASK_OUTPUT_BASE/$TASK_EXPERIMENT"
test -x "$TASK_PYTHON"
test -s "$TASK_IMPORTANCE_PATH"

export WANDB_PROJECT=sparse_embedding
export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0

nvidia-smi
sleep 3

"$TASK_PYTHON" -m accelerate.commands.launch train_compositional.py \
    --config_name "$TASK_MODEL_DIR" \
    --tokenizer_name "$TASK_MODEL_DIR" \
    --data_dir "$TASK_DATA_DIR" \
    --block_size 2048 \
    --preprocessing_num_workers 160 \
    --seed 42 \
    --bf16 \
    --ddp_timeout 21600 \
    --ddp_find_unused_parameters false \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
    --warmup_steps 500 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 250 \
    --dataloader_num_workers 8 \
    --report_to wandb \
    --output_dir "$TASK_OUTPUT_DIR" \
    --run_name "${TASK_EXPERIMENT}-qwen3-0.6b" \
    --arm product_code \
    --product_code_head_size "$TASK_HEAD_SIZE" \
    --product_code_num_hashes 4 \
    --product_code_num_buckets "$TASK_NUM_BUCKETS" \
    --product_code_assignment hashed \
    --product_code_importance_path "$TASK_IMPORTANCE_PATH" \
    --product_code_importance_key counts \
    --product_code_seed 0 \
    --tie_output
