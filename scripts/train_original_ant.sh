#!/bin/bash
# Train Qwen3-0.6B from scratch — Original ANT embedding

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
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --min_lr_rate 0.1 \
    --warmup_steps 500 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 250 \
    --dataloader_num_workers 8 \
    --report_to wandb \
    --output_dir /opt/dlami/nvme/sparse_emb_outputs/original_ant \
    --run_name original-ant-qwen3-0.6b \
    --arm original_ant \
    --K 4096 \
    --emb_lr 1e-2 \
    --lam 1e-3
