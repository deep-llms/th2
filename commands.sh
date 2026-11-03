#1 +30+a
#th2-wc69-a2-depth-partc-launch-20261103-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_depth_partc --output /mnt/local/_outputs/deep-llms_th2/wc69-a2-depth-partc-20261103-a01 --session a2_depth_partc_wc69_20261103_a01 --code-commit e2af735a59127f1116dbf8d671fe5aa95f1bb12f --plan resources/a2_depth_wc69_plan_20261103_a01.json
