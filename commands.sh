#1 +120+a
#restart-dummy
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

pkill -f dummy.py
sleep 10

nvidia-smi
sleep 3

python dummy.py
