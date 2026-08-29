#!/bin/bash
# B200-matched native dense tied Qwen3-0.6B baseline.
# This ablation intentionally omits --ddp_find_unused_parameters so that
# Transformers 5.9.0 selects its default behavior.
# Batch: 16/device x 4 accumulation x 8 GPUs = 512 sequences/update.

set -euo pipefail

if [ "${CONDA_DEFAULT_ENV:-}" != "sparse_emb" ]; then
    if [ -x /mnt/local/conda-py311/bin/conda ]; then
        eval "$(/mnt/local/conda-py311/bin/conda shell.bash hook)"
    elif [ -x /mnt/local/conda/bin/conda ]; then
        eval "$(/mnt/local/conda/bin/conda shell.bash hook)"
    else
        echo "ERROR: activate the sparse_emb conda environment before launching"
        exit 1
    fi
    conda activate sparse_emb
fi

TASK_PYTHON="${SPARSE_EMB_PYTHON:-$(command -v python3.11)}"
TASK_MODEL_DIR="${SPARSE_EMB_MODEL_DIR:-Qwen/Qwen3-0.6B}"
TASK_DATA_DIR="${SPARSE_EMB_DATA_DIR:-/mnt/local/_data/sparse_embedding/data/Qwen_Qwen3-0.6B/train}"
TASK_OUTPUT_BASE="${SPARSE_EMB_OUTPUT_BASE:-/mnt/local/_outputs/sparse_embedding}"
TASK_OUTPUT_DIR="$TASK_OUTPUT_BASE/dense_tied_baseline_b200_ddp_default"
test -x "$TASK_PYTHON"

"$TASK_PYTHON" - "$TASK_MODEL_DIR/config.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    config = json.load(handle)
assert config.get("tie_word_embeddings") is True, (
    "dense baseline requires native exact input/output tying",
    config.get("tie_word_embeddings"),
)
print("dense_baseline_config_tie_word_embeddings=true")
PY

export WANDB_PROJECT=sparse_embedding
export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0

nvidia-smi
sleep 3

"$TASK_PYTHON" -m accelerate.commands.launch train.py \
    --config_name "$TASK_MODEL_DIR" \
    --tokenizer_name "$TASK_MODEL_DIR" \
    --data_dir "$TASK_DATA_DIR" \
    --block_size 2048 \
    --preprocessing_num_workers 160 \
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
    --report_to wandb \
    --output_dir "$TASK_OUTPUT_DIR" \
    --run_name dense-tied-baseline-b200-ddp-default-qwen3-0.6b
