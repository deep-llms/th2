#1 +60+a
#th2-ohmv-multi-exit-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_multi_exit --commit 651173d8334f4ab964412c634663bb76112cc064 --prepared-sha256 a3995d1cce284d60f63519e08071609c1233d547d759a68c4138b27c60366f13
