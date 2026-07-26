#1
#train-ant-experiments
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

nvidia-smi
sleep 3

export WANDB_MODE=offline
python run_experiments.py --stop-at-step 10000 --log-dir /opt/dlami/nvme/sparse_emb_outputs/logs
