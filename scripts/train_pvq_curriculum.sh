#!/bin/bash
# Published P-VQ phase B: dense tied curriculum compression.
# Phase A is the existing dense tied baseline; compact conversion follows with
# scripts/convert_dense_baseline.py and phase C uses train_pvq_tied.sh.

set -euo pipefail

: "${PVQ_SOURCE_CHECKPOINT:?Set PVQ_SOURCE_CHECKPOINT to a dense tied checkpoint}"

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
TASK_OUTPUT_DIR="$TASK_OUTPUT_BASE/pvq_dense_curriculum_w768"
test -x "$TASK_PYTHON"
test -d "$PVQ_SOURCE_CHECKPOINT"

export WANDB_PROJECT=sparse_embedding
export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0

nvidia-smi
sleep 3

"$TASK_PYTHON" -m accelerate.commands.launch train.py \
    --model_name_or_path "$PVQ_SOURCE_CHECKPOINT" \
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
    --run_name pvq-dense-curriculum-w768-qwen3-0.6b \
    --pvq_curriculum \
    --pvq_shared_dim 768 \
    --pvq_k_begin 1024 \
    --pvq_k_end 128 \
    --pvq_k_decay 128 \
    --pvq_cluster_every 1000 \
    --pvq_curriculum_steps 10000 \
    --pvq_cluster_iters 100 \
    --pvq_cluster_restarts 10 \
    --pvq_cluster_device cuda
