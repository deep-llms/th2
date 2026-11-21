#1 +60+a
#th2-ohmv-adaptation-factors-gpu-20261121-a02
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_adaptation_factors --commit 959660a2882bdd5e55191f930bd1f084c7a535c8 --source-sha 0c6dc3331f6803b9277b8c3cc33b961e3e2e9c71401d144c334eb8346e856f29 --prepared-sha bcfd7f874322f905779c8523d91076cc1c8d83f724fb4fb476d3eae204be3c71
