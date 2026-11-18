#1 +60+a
#th2-ohmv-full-context-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_full_context --commit e99657b3378f60474c8ace2bc78d4ab2a9c0b6e8 --prepared-sha256 a4f5b5cefcbf5939ba6270ea0215e772373a358ae7be2a65b3bf91278fda2e11
