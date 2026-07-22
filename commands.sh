#1
#train-baseline
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

nvidia-smi
sleep 3

bash scripts/train_qwen3_0.6b_baseline.sh
