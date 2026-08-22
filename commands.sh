#1 +60+a
#th2-verify-qwen3-0.6b-download-20260822
set -euo pipefail

echo '=== th2 Qwen3-0.6B download verification ==='
date -u
hostname

TASK_MODEL_DIR="/mnt/local/_models/@PROJECT@/Qwen3-0.6B"
echo "model_dir=$TASK_MODEL_DIR"
test -d "$TASK_MODEL_DIR"

echo '=== model files ==='
ls -lah "$TASK_MODEL_DIR"
find "$TASK_MODEL_DIR" -maxdepth 2 -type f -printf '%s bytes  %p\n' | sort

for TASK_FILE in config.json model.safetensors tokenizer.json tokenizer_config.json; do
    test -s "$TASK_MODEL_DIR/$TASK_FILE"
done

python3 - "$TASK_MODEL_DIR/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)

assert config["model_type"] == "qwen3", config.get("model_type")
assert config["hidden_size"] == 1024, config.get("hidden_size")
assert config["num_hidden_layers"] == 28, config.get("num_hidden_layers")
assert config["vocab_size"] == 151936, config.get("vocab_size")
print("config_ok", {
    "model_type": config["model_type"],
    "hidden_size": config["hidden_size"],
    "num_hidden_layers": config["num_hidden_layers"],
    "vocab_size": config["vocab_size"],
})
PY

echo '=== weight checksum and disk usage ==='
sha256sum "$TASK_MODEL_DIR/model.safetensors"
du -sh "$TASK_MODEL_DIR"

echo 'TH2 QWEN3-0.6B DOWNLOAD VERIFY OK'
