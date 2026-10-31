#1 +30+a
#th2-6y3o-quant-qt3-reaudit-20261031-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u /mnt/local/deep-llms_th2/scripts/a2_quant_qt3_reaudit.py --plan /mnt/local/deep-llms_th2/resources/a2_quant_qt3_reaudit_20261031.json --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/qt3_revision_20261031_a01 --processes 12 2>&1 | tee /mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/qt3_revision_20261031_a01.log
