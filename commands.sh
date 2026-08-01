#1
#prepare-data
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

python prepare_data.py \
    --raw-dir /opt/dlami/nvme/sparse_emb_data/raw \
    --data-dir /opt/dlami/nvme/sparse_emb_data \
    --tokenizer-name Qwen/Qwen3-0.6B \
    --num-workers 4
