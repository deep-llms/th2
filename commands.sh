#1 +120+a
#th2-run-dummy
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

pkill -f dummy.py
sleep 10

nvidia-smi | head -12

python dummy.py
