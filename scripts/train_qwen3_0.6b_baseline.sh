#!/bin/bash
#
# Train Qwen3-0.6B from scratch (baseline) on multilingual data.
#
# Hardware: 8x H200 141GB
# Model: ~0.66B params
# Data: ~35B tokens (30B en + 5x1B other languages) x 1 epoch
# Effective batch: 16 * 4 * 8 = 512 sequences x 2048 = 1M tokens/step
# Total steps: ~35B / 1M ≈ 35,000 steps
# Warmup: 500 steps
# Min LR: 10% of max LR = 3e-5
#

export WANDB_PROJECT="sparse_embedding"
export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0

accelerate launch train.py \
    --config_name Qwen/Qwen3-0.6B \
    --tokenizer_name Qwen/Qwen3-0.6B \
    --data_dir /opt/dlami/nvme/sparse_emb_data/Qwen_Qwen3-0.6B/train \
    --block_size 2048 \
    --preprocessing_num_workers 160 \
    --output_dir /opt/dlami/nvme/sparse_emb_outputs/baseline \
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
    --run_name qwen3-0.6b-baseline \
    --report_to wandb
