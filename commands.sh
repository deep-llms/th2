#1
#install-lm-eval
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

pip install lm-eval[hf]
python -c "import lm_eval; print('lm_eval OK:', lm_eval.__version__)"
