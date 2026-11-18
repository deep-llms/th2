#1 +60+a
#th2-ohmv-warm-downstream-small-package-20261118-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PYCODE'
from efficient_llm.warm_order.data import sha
assert sha('efficient_llm/warm_order/downstream_pack.py')=='5aaeecd9111dae84f4bd3072e1566daa0b74bbefd6137ad290262df46103d751'
from efficient_llm.warm_order.downstream_pack import finalize
finalize()
PYCODE
