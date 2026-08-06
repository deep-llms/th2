#1
#check-eval-env
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3

conda env list
sleep 3

conda activate eval
sleep 3

python -c "import lm_eval; print('lm_eval OK:', lm_eval.__version__)"
