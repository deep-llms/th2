#1 +20+a
#th2-6y3o-a2-depth-partb-launch-20261101-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_depth_partb --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-depth-partb-20261101-a01 --session a2_depth_partb_20261101_a01 --code-commit a9cbd433d1fb91642c266319463e91c7a931afc4
