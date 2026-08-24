#1 +120+a
#th2-reverify-stop-burn-full-eval-recent-10k-20260824
#!/usr/bin/env bash
set -euo pipefail

TASK_PROJECT_DIR="$PWD"
TASK_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_BENCH_ROOT=/mnt/local/_data/@PROJECT@/benchmarks/hf
TASK_DATA_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/eval
TASK_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_INDEPENDENT_CKPT="$TASK_OUTPUT_BASE/lowrank_independent_output_r128/checkpoint-10000"
TASK_SHARED_LOCAL_CKPT="$TASK_OUTPUT_BASE/shared_local_tied_g16/checkpoint-10000"
TASK_LAUNCH_LOG="$TASK_OUTPUT_BASE/eval_parallel_independent_lr128_shared_local_g16_10k.log"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LM_EVAL_DATASET_ROOT="$TASK_BENCH_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

die() {
    echo "ERROR: $*"
    exit 1
}

proc_snapshot() {
    local pid=$1
    local process_stat process_rest
    [[ -r "/proc/$pid/stat" ]] || return 1
    process_stat=$(<"/proc/$pid/stat") || return 1
    [[ "$process_stat" == *') '* ]] || return 1
    process_rest=${process_stat#*) }
    TASK_PROC_STATE=${process_rest%% *}
    TASK_PROC_START_TICKS=$(awk '{print $20}' <<<"$process_rest")
    [[ "$TASK_PROC_START_TICKS" =~ ^[0-9]+$ ]] || return 1
}

same_proc() {
    local pid=$1
    local expected_start=$2
    proc_snapshot "$pid" || return 1
    [[ "$TASK_PROC_START_TICKS" == "$expected_start" ]]
    [[ "$TASK_PROC_STATE" != Z && "$TASK_PROC_STATE" != X ]]
}

echo '=== re-verify exact benchmark snapshots before touching GPU burns ==='
date -u
hostname
test -x "$TASK_EVAL_PYTHON"

declare -A TASK_EXPECTED_FILES=(
    [facebook/xnli]=108
    [facebook/belebele]=256
    [cambridgeltl/xcopa]=90
    [juletxara/xstory_cloze]=50
    [google-research-datasets/paws-x]=48
    [Rowan/hellaswag]=12
    [alexandrainst/m_hellaswag]=74
)
declare -A TASK_EXPECTED_BYTES=(
    [facebook/xnli]=1845652837
    [facebook/belebele]=251694558
    [cambridgeltl/xcopa]=1210541
    [juletxara/xstory_cloze]=10327779
    [google-research-datasets/paws-x]=67866844
    [Rowan/hellaswag]=36802833
    [alexandrainst/m_hellaswag]=925863657
)
declare -A TASK_EXPECTED_HASHES=(
    [facebook/xnli]=1fe8d392aa645ae4fe6264e7bf9ad9f71095cdb01e8b07eb543e2a6180dcd5e6
    [facebook/belebele]=4490323edfcf4e7c4df00aaa49b104635dca5d7f73f8f7dfdbe6d24d1fb2db06
    [cambridgeltl/xcopa]=6af0e4c3b9f3c563c97711eea91b7e89b1bb08793185398cf476815b4a2a6733
    [juletxara/xstory_cloze]=b0606cd565c3d80d974795a522e0795db0c6850c8ee6ec8df90caeac347c6585
    [google-research-datasets/paws-x]=489ea6ba31bd09e5b0002c0384dfdc27157ad4b55c9a6142944a9464a16c6e5e
    [Rowan/hellaswag]=81b9c31939730ef87c0bc06d8d0314ae1bb7be4b04778b426348e2d2148b0476
    [alexandrainst/m_hellaswag]=1564ac62ea98197611dcf4da4421ef0185221729391da9428b5b452ed562f4f0
)
TASK_DATASET_DIRS=(
    facebook/xnli
    facebook/belebele
    cambridgeltl/xcopa
    juletxara/xstory_cloze
    google-research-datasets/paws-x
    Rowan/hellaswag
    alexandrainst/m_hellaswag
)

for relpath in "${TASK_DATASET_DIRS[@]}"; do
    dataset_dir="$TASK_BENCH_ROOT/$relpath"
    test -d "$dataset_dir"
    file_count=$(find "$dataset_dir" -type f | wc -l)
    byte_count=$(find "$dataset_dir" -type f -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')
    tree_hash=$(find "$dataset_dir" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
    [[ "$file_count" == "${TASK_EXPECTED_FILES[$relpath]}" ]] \
        || die "$relpath file count changed: $file_count"
    [[ "$byte_count" == "${TASK_EXPECTED_BYTES[$relpath]}" ]] \
        || die "$relpath byte count changed: $byte_count"
    [[ "$tree_hash" == "${TASK_EXPECTED_HASHES[$relpath]}" ]] \
        || die "$relpath content hash changed: $tree_hash"
    printf 'VERIFIED %-45s files=%-5s bytes=%-12s sha256_tree=%s\n' \
        "$relpath" "$file_count" "$byte_count" "$tree_hash"
done

"$TASK_EVAL_PYTHON" - <<'PY'
from pathlib import Path

from datasets import load_dataset

root = Path('/mnt/local/_data/@PROJECT@/benchmarks/hf')
cases = [
    *[("facebook/xnli", lang, "validation", f"xnli_{lang}")
      for lang in ("en", "vi", "zh", "ru", "de", "ar")],
    *[("facebook/belebele", lang, "test", f"belebele_{lang}")
      for lang in ("eng_Latn", "vie_Latn", "zho_Hans", "rus_Cyrl", "deu_Latn", "arb_Arab")],
    *[("cambridgeltl/xcopa", lang, "validation", f"xcopa_{lang}")
      for lang in ("vi", "zh")],
    *[("juletxara/xstory_cloze", lang, "eval", f"xstorycloze_{lang}")
      for lang in ("en", "ar", "ru", "zh")],
    *[("google-research-datasets/paws-x", lang, "validation", f"paws_{lang}")
      for lang in ("en", "de", "zh")],
    ("Rowan/hellaswag", None, "validation", "hellaswag"),
    *[("alexandrainst/m_hellaswag", lang, "val", f"hellaswag_{lang}")
      for lang in ("ar", "de", "ru", "vi")],
]
assert len(cases) == 26
for relpath, config, split, task in cases:
    dataset = load_dataset(
        str(root / relpath), name=config, split=split, streaming=True
    )
    row = next(iter(dataset))
    assert isinstance(row, dict) and row, (task, row)
    print(f"OFFLINE_LOAD_OK task={task} columns={len(row)}")
print('BENCHMARK REVERIFICATION OK: 7 repositories, 26 tasks')
PY

echo '=== validate and install offline lm-eval task paths ==='
cd "$TASK_PROJECT_DIR"
"$TASK_EVAL_PYTHON" - <<'PY'
import os

import lm_eval

from eval.benchmarks import TASK_CONFIGS, _DATASET_PATH_PATCHES, patch_lm_eval_dataset_paths

root = os.environ['LM_EVAL_DATASET_ROOT']
patch_lm_eval_dataset_paths(root)
tasks_dir = os.path.join(os.path.dirname(lm_eval.__file__), 'tasks')
for relative_file, repository, _aliases in _DATASET_PATH_PATCHES:
    expected = f"dataset_path: {os.path.join(root, repository)}"
    with open(os.path.join(tasks_dir, relative_file), encoding='utf-8') as handle:
        lines = [line.strip() for line in handle if line.startswith('dataset_path:')]
    assert lines == [expected], (relative_file, lines, expected)
tasks = [task for group in TASK_CONFIGS.values() for task in group]
assert len(tasks) == len(set(tasks)) == 26
print('LM_EVAL LOCAL PATH PATCH OK: 10 task configs, 26 unique tasks')
PY

echo '=== identify the exact eight supervised GPU burn workers ==='
mapfile -t TASK_GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader \
        | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
)
[[ "${#TASK_GPU_NAMES[@]}" -eq 8 ]] || die "expected 8 GPUs, found ${#TASK_GPU_NAMES[@]}"
for index in "${!TASK_GPU_NAMES[@]}"; do
    [[ "${TASK_GPU_NAMES[$index]}" == *B200* ]] \
        || die "GPU $index is not a B200: ${TASK_GPU_NAMES[$index]}"
done

mapfile -t TASK_BURN_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
[[ "${#TASK_BURN_PIDS[@]}" -eq 8 ]] \
    || die "expected exactly 8 GPU burn PIDs, found ${#TASK_BURN_PIDS[@]}: ${TASK_BURN_PIDS[*]}"

declare -A TASK_BURN_START_TICKS=()
for pid in "${TASK_BURN_PIDS[@]}"; do
    proc_snapshot "$pid" || die "cannot inspect GPU PID $pid"
    [[ "$TASK_PROC_STATE" != Z && "$TASK_PROC_STATE" != X ]] \
        || die "GPU PID $pid is not live"
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    [[ "$cmdline" == *scripts/gpu_burn.py* ]] \
        || die "refusing to kill non-burn GPU PID $pid: $cmdline"
    TASK_BURN_START_TICKS[$pid]=$TASK_PROC_START_TICKS
    echo "validated_burn_pid=$pid start_ticks=$TASK_PROC_START_TICKS argv=$cmdline"
done

echo '=== stop only the validated GPU burn workers ==='
for pid in "${TASK_BURN_PIDS[@]}"; do
    same_proc "$pid" "${TASK_BURN_START_TICKS[$pid]}" \
        || die "burn PID $pid changed identity before TERM"
    kill -TERM "$pid"
done

for attempt in $(seq 1 30); do
    survivors=()
    for pid in "${TASK_BURN_PIDS[@]}"; do
        same_proc "$pid" "${TASK_BURN_START_TICKS[$pid]}" && survivors+=("$pid")
    done
    [[ "${#survivors[@]}" -eq 0 ]] && break
    sleep 2
done
if [[ "${#survivors[@]}" -gt 0 ]]; then
    echo "TERM timeout; sending KILL to validated survivors: ${survivors[*]}"
    for pid in "${survivors[@]}"; do
        same_proc "$pid" "${TASK_BURN_START_TICKS[$pid]}" && kill -KILL "$pid"
    done
fi

for attempt in $(seq 1 30); do
    mapfile -t TASK_GPU_PIDS < <(
        nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
            | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
    )
    [[ "${#TASK_GPU_PIDS[@]}" -eq 0 ]] && break
    sleep 2
done
[[ "${#TASK_GPU_PIDS[@]}" -eq 0 ]] \
    || die "GPU compute processes remain after stopping burns: ${TASK_GPU_PIDS[*]}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
echo 'TH2 ALL 8 B200 GPUS FREE; GPU BURNS STOPPED'

echo '=== full evaluation preflight ==='
test -s "$TASK_MODEL_DIR/config.json"
test -s "$TASK_MODEL_DIR/tokenizer.json"
for lang in en vi zh ru de ar; do
    test -d "$TASK_DATA_DIR/$lang"
    test "$(find "$TASK_DATA_DIR/$lang" -type f -name '*.arrow' | wc -l)" -gt 0
done
for checkpoint in "$TASK_INDEPENDENT_CKPT" "$TASK_SHARED_LOCAL_CKPT"; do
    test -d "$checkpoint"
    test -s "$checkpoint/config.json"
    test -s "$checkpoint/embedding.pt"
    test -s "$checkpoint/trainer_state.json"
    test -s "$checkpoint/model.safetensors"
done
test -s "$TASK_INDEPENDENT_CKPT/output_head.pt"

for artifact in \
    "$TASK_INDEPENDENT_CKPT/eval.log" \
    "$TASK_INDEPENDENT_CKPT/eval_ppl.json" \
    "$TASK_INDEPENDENT_CKPT/eval_benchmarks.json" \
    "$TASK_SHARED_LOCAL_CKPT/eval.log" \
    "$TASK_SHARED_LOCAL_CKPT/eval_ppl.json" \
    "$TASK_SHARED_LOCAL_CKPT/eval_benchmarks.json" \
    "$TASK_LAUNCH_LOG"; do
    [[ ! -e "$artifact" ]] || die "refusing to overwrite evaluation artifact: $artifact"
done

echo '=== launch full PPL + all 26 benchmarks for both checkpoint-10000 models ==='
"$TASK_EVAL_PYTHON" -u eval/eval_parallel.py \
    --checkpoints "$TASK_INDEPENDENT_CKPT" "$TASK_SHARED_LOCAL_CKPT" \
    --eval-dir "$TASK_DATA_DIR" \
    --tokenizer-name "$TASK_MODEL_DIR" \
    --bf16 \
    --num-gpus 8 \
    --log "$TASK_LAUNCH_LOG"

echo '=== verify complete evaluation outputs ==='
"$TASK_EVAL_PYTHON" - "$TASK_INDEPENDENT_CKPT" "$TASK_SHARED_LOCAL_CKPT" <<'PY'
import json
import math
import os
import sys

expected_languages = {'en', 'vi', 'zh', 'ru', 'de', 'ar'}
for checkpoint in sys.argv[1:]:
    with open(os.path.join(checkpoint, 'eval_ppl.json'), encoding='utf-8') as handle:
        perplexity = json.load(handle)
    with open(os.path.join(checkpoint, 'eval_benchmarks.json'), encoding='utf-8') as handle:
        benchmarks = json.load(handle)
    assert expected_languages == set(perplexity), (checkpoint, perplexity.keys())
    for language, metrics in perplexity.items():
        assert int(metrics['num_tokens']) > 0, (checkpoint, language, metrics)
        assert math.isfinite(float(metrics['loss'])), (checkpoint, language, metrics)
        assert math.isfinite(float(metrics['perplexity'])), (checkpoint, language, metrics)
    assert len(benchmarks) == 26, (checkpoint, len(benchmarks))
    for task, metrics in benchmarks.items():
        accuracy = metrics.get('acc,none', metrics.get('acc'))
        assert accuracy is not None and math.isfinite(float(accuracy)), (
            checkpoint, task, metrics
        )
    print(f"EVAL_JSON_OK checkpoint={checkpoint} ppl_languages=6 benchmark_tasks=26")
PY

grep -F 'Loaded compositional model: arm=lowrank' "$TASK_INDEPENDENT_CKPT/eval.log"
grep -F 'Loaded compositional model: arm=shared_local' "$TASK_SHARED_LOCAL_CKPT/eval.log"
if grep -HniE 'Traceback \(most recent call last\)|CUDA out of memory|FAILED \(code|Error:' \
    "$TASK_INDEPENDENT_CKPT/eval.log" "$TASK_SHARED_LOCAL_CKPT/eval.log"; then
    die 'failure signature found in an evaluation log'
fi

mapfile -t TASK_FINAL_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
[[ "${#TASK_FINAL_GPU_PIDS[@]}" -eq 0 ]] \
    || die "GPU processes remain after evaluation: ${TASK_FINAL_GPU_PIDS[*]}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
tail -80 "$TASK_LAUNCH_LOG"
echo 'TH2 FULL EVAL VERIFIED: BOTH RECENT CHECKPOINT-10000 MODELS COMPLETE; GPUS FREE'
