#!/bin/bash
#
# Train Qwen3-0.6B from scratch WITHOUT EmbHub (baseline) on multilingual data.
#
# Hardware: 8x H200 141GB
# Model: ~0.66B params
# Data: ~31.5B tokens (30B en + 5x300M other languages) x 1 epoch = ~31.5B tokens
# Effective batch: 16 * 4 * 8 = 512 sequences x 2048 = 1M tokens/step
# Total steps: ~31.5B / 1M ≈ 31,500 steps
# Warmup: 500 steps
# Min LR: 10% of max LR = 3e-5
#


export WANDB_PROJECT="cross_lingual_embedding_hub"
export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0

accelerate launch smoke_train.py \
    --config_name Qwen/Qwen3-0.6B \
    --tokenizer_name Qwen/Qwen3-0.6B \
    --data_dir /opt/dlami/nvme/embhub_data/Qwen_Qwen3-0.6B/train \
    --block_size 2048 \
    --preprocessing_num_workers 160 \
    --no_embhub \
    --output_dir /opt/dlami/nvme/smoke_test_outputs/baseline \
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
    --report_to wandb
