#1 +20+a
#th2-6y3o-a2-depth-partc-launch-20261101-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_depth_partc --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-depth-partc-20261101-a02 --session a2_depth_partc_20261101_a02 --code-commit dd55c8ba96eafabfed3de63a94e249da674c4f19 --plan resources/a2_depth_partc_plan_20261101.json
