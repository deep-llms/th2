#1 +60+a
#th2-ohmv-layer-adaptation-gpu-gate-20261121-a02
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONPATH="$PWD:$PWD/tests/warm_order"
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
"$TASK_PYTHON" -m unittest discover -s tests/warm_order -p 'test_adaptation*.py' -v
"$TASK_PYTHON" -m scripts.launch_layer_adaptation_gate --commit 88bab1f990edddbb719b2a71d2d11effab0cfd35 --prepared-sha256 67fcf590bb4386fb6490ef7a5e0fada2c9312066e39094bea058cd7885c0192e --source-sha256 2f86743c0249af80ed8034a878ac81b3c6a21229bb113370a0f4fd5ac16db938
