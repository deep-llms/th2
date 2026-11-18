#1 +60+a
#th2-ohmv-budget-refinement-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_budget_refinement --commit dca57b9f321bbea7ce8df26156bed75efd27f8c7 --prepared-sha256 88f8f81a142ef387486b0b1304ab493959ca475c3143c7222084121bba394968
