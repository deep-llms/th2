#1 +60+a
#th2-ohmv-restart-factorial-launch-20261120-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_restart_factorial --commit e647a82651dc239cb482e8dd8504abd57dc8192f --prepared-sha256 3eb8d92fe585ea7429aaa0b65bc51a8edce6c4be20a2ec330f8c6053f4c4113b
