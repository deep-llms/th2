#1 +60+a
#th2-ohmv-adaptation-context-gpu-20261121-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
TASK_COMMIT=$(git rev-parse HEAD)
"$TASK_PYTHON" -m scripts.launch_adaptation_context --commit "$TASK_COMMIT" --source-sha 31f9afe17b6790303f0cd8bf43d9bd0fbc33ef93cf6330fb7d97b876c15dff2f --prepared-sha 6d45f830b362be95e8132ca7c8b5093991f0aaab328f1a87932597c359fd06ad
