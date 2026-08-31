#!/bin/bash
# Parameter-matched tied residual subspace experts (global R120 + 12 R80 residuals).

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
TASK_OUTPUT_DIR="$TASK_OUTPUT_BASE/residual_subspace_experts_tied_g12_r120_q80"
TASK_PER_DEVICE_BATCH="${RSE_PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
TASK_GRADIENT_ACCUMULATION="${RSE_GRADIENT_ACCUMULATION_STEPS:-4}"
test -x "$TASK_PYTHON"
case "$TASK_PER_DEVICE_BATCH:$TASK_GRADIENT_ACCUMULATION" in
    *[!0-9:]*|0:*|*:0) echo "ERROR: RSE batch/accumulation must be positive integers" >&2; exit 1 ;;
esac
# Every registered production setting must preserve the B200 comparison's
# global batch: 8 GPUs x per-device batch x accumulation = 512.
test $((8 * TASK_PER_DEVICE_BATCH * TASK_GRADIENT_ACCUMULATION)) -eq 512

export WANDB_PROJECT=sparse_embedding
export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
    --per_device_train_batch_size "$TASK_PER_DEVICE_BATCH" \
    --gradient_accumulation_steps "$TASK_GRADIENT_ACCUMULATION" \
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
    --run_name residual-subspace-experts-tied-g12-r120-q80-qwen3-0.6b \
    --arm residual_subspace_experts \
    --rse_base_rank 120 \
    --rse_expert_rank 80 \
    --rse_num_experts 12 \
    --rse_router_dim 32 \
    --rse_top_k 2 \
    --rse_router_temperature 1.0 \
    --lambda_div 0.01 \
    --tie_output
