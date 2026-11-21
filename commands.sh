#1 +60+a
#th2-ohmv-adaptation-directions-gpu-20261121-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_adaptation_directions --commit 4946eefe3d4fe63d08d5cbaa8221d250db33068f --source-sha ef28062099b16e4a72f531a660791ab4c576379a726ad687094c4fb1c408d6fe --prepared-sha 45c9ec81d55d7af825f47f50426357346d6526424da2fa796edcd23f864c17e1
