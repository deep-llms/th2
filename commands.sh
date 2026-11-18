#1 +60+a
#th2-ohmv-causal-cost-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_causal_cost --commit 487eba03882b8c7bca83cfff86be3cb62e697055 --prepared-sha256 5b206714c1cd7bdedec2359487ea04be0f63cac27f433cd6ccadd870553a0c63
