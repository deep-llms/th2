#!/bin/bash
# Train Qwen3-0.6B from scratch — V0 embedding (static selection + anchor SAT)

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

export WANDB_PROJECT="sparse_embedding"
export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0

nvidia-smi
sleep 3

accelerate launch train_compositional.py \
    --config_name Qwen/Qwen3-0.6B \
    --tokenizer_name Qwen/Qwen3-0.6B \
    --data_dir /opt/dlami/nvme/sparse_emb_data/Qwen_Qwen3-0.6B/train \
    --block_size 2048 \
    --preprocessing_num_workers 160 \
    --seed 42 \
    --bf16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
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
    --output_dir /opt/dlami/nvme/sparse_emb_outputs/v0_post \
    --run_name v0-post-qwen3-0.6b \
    --arm v0 \
    --K 4096 \
    --d_x 128 \
    --d_k 64 \
    --gamma 1.0 \
    --max_k 16 \
    --v0_mode post
