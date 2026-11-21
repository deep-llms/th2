#1 +60+a
#th2-ohmv-layer-adaptation-gpu-gate-20261121-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD:$PWD/tests/warm_order"
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
"$TASK_PYTHON" -m unittest discover -s tests/warm_order -p 'test_adaptation*.py' -v
"$TASK_PYTHON" -m scripts.launch_layer_adaptation_gate --commit faba096d04f340ce6cf450c6e4a84941ea5e1ab3 --prepared-sha256 67fcf590bb4386fb6490ef7a5e0fada2c9312066e39094bea058cd7885c0192e --source-sha256 7d47c366f81ac2cdf26b7e2568f1efd160aad9a68df616120798df4284a2cddb
