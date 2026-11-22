#1 +60+a
#th2-ohmv-attention-shift-gpu-20261122-a02
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_attention_shift --commit 01406e854140cb9288ec27214c713256336ef93a --source-sha a72c613e71b90bc3a7d97022bd850af2f3804823f66bc35826b4bf68d9687fe3 --prepared-sha 4eaff197bdc24a4ebf92fd55a0f7ad6721493793169f28b163ce3a779e3d9fcf
