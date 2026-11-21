#1 +60+a
#th2-ohmv-path-background-gpu-20261121-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH="$PWD:$PWD/tests/warm_order"
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
"$TASK_PYTHON" -m unittest discover -s tests/warm_order -p 'test_path_background*.py' -v
"$TASK_PYTHON" -m scripts.launch_path_background --commit fb922bfd9135f77a8bd7ac69ba0322c7fa9c3427 --source-sha ef9ff11350d8691c98e234ffeb350dd24e19303d4b618609e904c4f5b15f16fb --prepared-sha 7fa46c15b3862483e36654fdb207d042b933a44a5f0bc4e7a9c3f2a7ee9b1918
