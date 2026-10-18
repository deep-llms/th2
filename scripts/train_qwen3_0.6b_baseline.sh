#!/usr/bin/env bash
# Example Qwen3-0.6B scratch recipe; all assets and GPU allocation are explicit.
# At eight ranks: 16 * 4 * 8 * 2048 = 1,048,576 input tokens/update.
# Configure/verify GPU ownership and Accelerate before invoking this script.
set -euo pipefail
if (( $# < 5 )); then
    echo "Usage: $0 ACCELERATE_CONFIG MODEL_CONFIG TOKENIZER_DIR DATA_DIR OUTPUT_DIR [Trainer arguments...]" >&2
    exit 2
fi
TASK_ACCELERATE_CONFIG=$1
TASK_MODEL_CONFIG=$2
TASK_TOKENIZER_DIR=$3
TASK_DATA_DIR=$4
TASK_OUTPUT_DIR=$5
shift 5
TASK_PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
test -f "$TASK_ACCELERATE_CONFIG"
test -e "$TASK_MODEL_CONFIG"
test -d "$TASK_TOKENIZER_DIR"
test -e "$TASK_DATA_DIR"

export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# Select an activated ML environment or set PYTHON_BIN to its interpreter.
# Extra Trainer arguments can override this example's defaults.
exec "${PYTHON_BIN:-python}" -m accelerate.commands.launch \
    --config_file "$TASK_ACCELERATE_CONFIG" --mixed_precision bf16 \
    "$TASK_PROJECT_DIR/train.py" \
    --config_name "$TASK_MODEL_CONFIG" \
    --tokenizer_name "$TASK_TOKENIZER_DIR" \
    --data_dir "$TASK_DATA_DIR" \
    --output_dir "$TASK_OUTPUT_DIR" \
    --block_size 2048 \
    --document_end_token '<|endoftext|>' \
    --seed 42 \
    --bf16 \
    --ddp_timeout 21600 \
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
    --run_name qwen3-0.6b-scratch-baseline \
    --report_to none \
    "$@"
