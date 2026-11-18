#1 +60+a
#th2-ohmv-recursive-depth-launch-20261119-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.launch_recursive_depth --commit 4d1562c48c8bde0162fcbb707f1f5c047382308c --prepared-sha256 61ad351e94dccb6a382cc1d47bc05e0808175a3bb281ad59df5455ce955c66b9
