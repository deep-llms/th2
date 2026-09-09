#1 +120+a
#th2-swt-train-next-capacity-14arms-5k-s42-20260909-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt
test "$CONDA_DEFAULT_ENV" = swt
TASK_PYTHON=/mnt/local/conda-py311/envs/swt/bin/python
test "$(command -v python)" = "$TASK_PYTHON"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline

# Pin the reviewed source used by the successful eight-B200 production smoke.
echo 'cdaedf19db69c35a4bf58bef8a1b3224647b17359c0fa098d02277df2b26341c  capacity_allocation/modeling.py' | sha256sum -c -
echo '76b5158b9e30cf199cf47d7521ac6e3db4549bbfab88fdee12a95b35ee89ddb0  scripts/train_capacity_b200.sh' | sha256sum -c -
echo 'edc67b8db554a8053f8e183b3108ad643e4fe98602577a190911bd27b6b9a93f  resources/accelerate_config.yaml' | sha256sum -c -
echo '2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174  resources/llm_pretrain_burn.py' | sha256sum -c -

# CPU-only final topology/equation/count regression; existing burns remain untouched.
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" -m unittest tests.test_capacity_models -v

TASK_RUN_ROOT=/mnt/local/_outputs/deep-llms_th2/swt/next_capacity_14arms_5k_s42_20260909_a01
test ! -e "$TASK_RUN_ROOT"
SWT_STOP_AT_STEP=5000 bash scripts/train_capacity_b200.sh "$TASK_RUN_ROOT" \
  T768 P512-128-384 FixedResidual WNW \
  A640 A768 A768-Direct \
  D-1024 O1024-I232 O1024-I256 \
  C-Direct D-Direct T512 O1280
