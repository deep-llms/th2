#1 +60+a
#th2-ccm-prepare-real-smoke-20260913-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-8mgy-worker-0
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate train_env
test "$CONDA_DEFAULT_ENV" = train_env
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
TASK_ASSETS=/mnt/local/_models/@PROJECT@/Qwen3-0.6B-Base-da87bfb608c14b7cf20ba1ce41287e8de496c0cd
TASK_DATA=/mnt/local/_data/@PROJECT@/ccm/smoke_20260913_a01
TASK_OUTPUT=/mnt/local/_outputs/@PROJECT@/ccm_smoke_20260913_a01
test ! -e "$TASK_DATA"
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_DATA" "$TASK_OUTPUT"
python -m ccm prepare --raw-dir /mnt/local/_data/@PROJECT@/data/raw --source-manifest resources/culturax_raw_manifest.tsv --dataset-revision 6a8734bc69fefcbb7735f4f9250f43e4cd7a442e --tokenizer-path "$TASK_ASSETS" --tokenizer-manifest resources/qwen3_base_assets.json --budget resources/real_smoke_budget.json --engineering --output "$TASK_DATA/corpus"
python -m ccm validate-data --data "$TASK_DATA/corpus" > "$TASK_OUTPUT/data_validation.json"
python -m ccm vocabulary --data "$TASK_DATA/corpus" --output "$TASK_DATA/vocabulary.npz"
python -m ccm coverage --data "$TASK_DATA/corpus" --vocabulary "$TASK_DATA/vocabulary.npz" --output "$TASK_OUTPUT/coverage.json"
python - "$TASK_ASSETS" "$TASK_DATA/model_config" <<'PY'
import sys
from ccm.model import pilot_config
from ccm.cli import asset_identity
assets=asset_identity(sys.argv[1],'resources/qwen3_base_assets.json')
config=pilot_config(sys.argv[1])
assert config.num_hidden_layers==12
config.save_pretrained(sys.argv[2])
print('PINNED_BASE_ASSETS_AND_12_LAYER_CONFIG_VERIFIED',assets['revision'],flush=True)
PY
cat "$TASK_OUTPUT/data_validation.json" "$TASK_OUTPUT/coverage.json"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
date -u
echo CCM_REAL_DATA_PREPARATION_COMPLETE
