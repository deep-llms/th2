#1 +60+a
#th2-ohmv-strength-refinement-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_strength_refinement --commit 409da410bd30467a3b05fc22417541f65f1dcd86 --prepared-sha256 d26ca3fca8d74fc995c7fcb1ed7c7109f05eb0b303dc00b0774b30a0100c7b55
