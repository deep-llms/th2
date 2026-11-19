#1 +60+a
#th2-ohmv-pair-order-launch-20261120-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_pair_order --commit 880a7a867627cf22400c7e407441856dd38b61c8 --prepared-sha256 fb6d23d144fdeb6081465730e746f40c328892a583c62e4bbce9a35df4173472
