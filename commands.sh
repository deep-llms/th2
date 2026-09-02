#1 +120+a
#th2-sample-verified-culturax-offline-fresh-pod-20260902-a19
set -euo pipefail

echo '=== sample verified CulturaX data offline ==='
date -u
hostname

TASK_PROJECT_DIR="$PWD"
TASK_CONDA=/mnt/local/conda-py311/bin/conda
TASK_DATA_ROOT=/mnt/local/_data/@PROJECT@/data
TASK_RAW_DIR="$TASK_DATA_ROOT/raw"
TASK_OUTPUT_ROOT="$TASK_DATA_ROOT/Qwen_Qwen3-0.6B"
TASK_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_MANIFEST="$TASK_PROJECT_DIR/resources/culturax_raw_manifest.tsv"
TASK_PREPARE_SCRIPT="$TASK_PROJECT_DIR/prepare_data.py"

test -x "$TASK_CONDA"
eval "$("$TASK_CONDA" shell.bash hook)"
conda activate sparse_emb
TASK_PYTHON="$(command -v python3.11)"
echo "conda_env=$CONDA_DEFAULT_ENV"
echo "python=$TASK_PYTHON"
test "$CONDA_DEFAULT_ENV" = sparse_emb
test "$TASK_PYTHON" = /mnt/local/conda-py311/envs/sparse_emb/bin/python3.11

for TASK_REQUIRED_DIR in "$TASK_DATA_ROOT" "$TASK_RAW_DIR" "$TASK_MODEL_DIR"; do
    test -d "$TASK_REQUIRED_DIR"
done
for TASK_REQUIRED_FILE in \
    "$TASK_MODEL_DIR/config.json" \
    "$TASK_MODEL_DIR/tokenizer.json" \
    "$TASK_MODEL_DIR/tokenizer_config.json" \
    "$TASK_MANIFEST" \
    "$TASK_PREPARE_SCRIPT"; do
    test -s "$TASK_REQUIRED_FILE"
done

case "$TASK_OUTPUT_ROOT" in
    /mnt/local/_data/*/data/Qwen_Qwen3-0.6B) ;;
    *) echo "REFUSE: unexpected output path: $TASK_OUTPUT_ROOT"; exit 1 ;;
esac
if [ -e "$TASK_OUTPUT_ROOT" ]; then
    echo "REFUSE: sampled output already exists: $TASK_OUTPUT_ROOT"
    du -sh "$TASK_OUTPUT_ROOT" || true
    exit 1
fi

echo '=== storage preflight ==='
df -h "$TASK_DATA_ROOT"
TASK_AVAILABLE_BYTES="$(df --output=avail -B1 "$TASK_DATA_ROOT" | tail -n 1 | tr -d ' ')"
echo "available_bytes=$TASK_AVAILABLE_BYTES"
test "$TASK_AVAILABLE_BYTES" -ge 200000000000

echo '=== verify local model config and tokenizer ==='
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=''
"$TASK_PYTHON" - "$TASK_MODEL_DIR" <<'PY'
import json
import sys

import datasets
import pyarrow
import transformers
from transformers import AutoTokenizer

model_dir = sys.argv[1]
with open(f"{model_dir}/config.json", encoding="utf-8") as handle:
    config = json.load(handle)
assert config["model_type"] == "qwen3", config.get("model_type")
assert config["hidden_size"] == 1024, config.get("hidden_size")
assert config["num_hidden_layers"] == 28, config.get("num_hidden_layers")
assert config["vocab_size"] == 151936, config.get("vocab_size")
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
ids = tokenizer("offline tokenizer check", add_special_tokens=False)["input_ids"]
assert ids
print(f"python={sys.version.split()[0]}")
print(f"datasets={datasets.__version__} pyarrow={pyarrow.__version__} transformers={transformers.__version__}")
print(f"tokenizer_class={type(tokenizer).__name__} tokenizer_test_tokens={len(ids)}")
print("LOCAL_QWEN3_TOKENIZER_OK")
PY

echo '=== validate exact local six-language sample plan ==='
"$TASK_PYTHON" -u "$TASK_PREPARE_SCRIPT" sample \
    --dry-run \
    --raw-dir "$TASK_RAW_DIR" \
    --data-dir "$TASK_DATA_ROOT" \
    --manifest "$TASK_MANIFEST" \
    --tokenizer-name Qwen/Qwen3-0.6B \
    --tokenizer-path "$TASK_MODEL_DIR" \
    --local-files-only

echo '=== run offline sampling for en vi zh ru de ar ==='
"$TASK_PYTHON" -u "$TASK_PREPARE_SCRIPT" sample \
    --raw-dir "$TASK_RAW_DIR" \
    --data-dir "$TASK_DATA_ROOT" \
    --manifest "$TASK_MANIFEST" \
    --tokenizer-name Qwen/Qwen3-0.6B \
    --tokenizer-path "$TASK_MODEL_DIR" \
    --local-files-only \
    --langs en vi zh ru de ar \
    --flush-every 1 \
    --tokenize-batch-size 4096

echo '=== verify sampled H100-compatible output layout ==='
for TASK_LANG in en vi zh ru de ar; do
    TASK_TRAIN_DIR="$TASK_OUTPUT_ROOT/train/$TASK_LANG"
    TASK_EVAL_DIR="$TASK_OUTPUT_ROOT/eval/$TASK_LANG"
    test -d "$TASK_TRAIN_DIR"
    test -d "$TASK_EVAL_DIR"
    test -s "$TASK_EVAL_DIR/dataset_info.json"
    test -s "$TASK_EVAL_DIR/state.json"
    TASK_SHARD_COUNT="$(find "$TASK_TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'shard_*' | wc -l)"
    TASK_TRAIN_ARROW_COUNT="$(find "$TASK_TRAIN_DIR" -type f -name '*.arrow' | wc -l)"
    TASK_EVAL_ARROW_COUNT="$(find "$TASK_EVAL_DIR" -maxdepth 1 -type f -name '*.arrow' | wc -l)"
    echo "$TASK_LANG train_shards=$TASK_SHARD_COUNT train_arrow_files=$TASK_TRAIN_ARROW_COUNT eval_arrow_files=$TASK_EVAL_ARROW_COUNT"
    test "$TASK_SHARD_COUNT" -gt 0
    test "$TASK_TRAIN_ARROW_COUNT" -gt 0
    test "$TASK_EVAL_ARROW_COUNT" -gt 0
done

du -sh "$TASK_OUTPUT_ROOT"
df -h "$TASK_DATA_ROOT"
echo '=== GPU burns remain active after CPU-only sampling ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits
echo 'TH2 OFFLINE CULTURAX SAMPLING OK'
