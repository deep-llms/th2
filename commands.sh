#1
#th2-eval-lowrank-retry
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate eval
sleep 3

# Clean partial eval.log files from the killed eval run
for d in /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-*/; do
    rm -f "$d/eval.log"
done
echo "cleaned partial eval.log files"

nvidia-smi | head -12

python eval/eval_parallel.py \
    --checkpoints \
        /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-1000 \
        /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-2000 \
        /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-3000 \
        /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-4000 \
        /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-5000 \
        /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-6000 \
        /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-7000 \
        /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-8000 \
        /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-9000 \
        /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-10000 \
    --eval-dir /opt/dlami/nvme/sparse_emb_data/Qwen_Qwen3-0.6B/eval \
    --tokenizer-name Qwen/Qwen3-0.6B \
    --bf16 \
    --num-gpus 8
