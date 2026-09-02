#!/usr/bin/env bash
set -euo pipefail

TASK_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TASK_PROJECT_NAME="$(basename "$TASK_PROJECT")"
TASK_PYTHON=''
TASK_PYTHON_CANDIDATES=(
    /mnt/local/conda-py311/envs/eval/bin/python3.11
    /mnt/local/conda-py311/envs/eval/bin/python
    /mnt/local/conda/envs/eval/bin/python3.11
    /mnt/local/conda/envs/eval/bin/python
    "$HOME/miniconda3/envs/eval/bin/python3.11"
    "$HOME/miniconda3/envs/eval/bin/python"
)
TASK_BURN_PYTHON=/usr/bin/python3
TASK_BURN_SCRIPT=/tmp/llm_pretrain_burn.py
TASK_BURN_SHA=3cdcc857bd01b096e20a02640fa85f0b8be7607e3c2b22a89a704bbac3650857
TASK_OUTPUT_BASE="/mnt/local/_outputs/$TASK_PROJECT_NAME"
TASK_DATA_DIR="/mnt/local/_data/$TASK_PROJECT_NAME/data/Qwen_Qwen3-0.6B/eval"
TASK_TOKENIZER="/mnt/local/_models/$TASK_PROJECT_NAME/Qwen3-0.6B"
TASK_HASHED_ROOT="$TASK_OUTPUT_BASE/product_code_hashed_h2048"
TASK_HASHED_CKPT="$TASK_HASHED_ROOT/checkpoint-10000"
TASK_CONTROL_ROOT="$TASK_OUTPUT_BASE/groupreduce_matched_nested_tied_t4"
TASK_REFERENCE_ROOT="$TASK_OUTPUT_BASE/frequency_binned_ppl_four_10k_20260831_a03/merged"
TASK_DENSE_BYTOKEN="$TASK_REFERENCE_ROOT/dense_ddp_false_eval_ppl_bytoken.npz"
TASK_CONTROL_BYTOKEN="$TASK_REFERENCE_ROOT/groupreduce_t4_eval_ppl_bytoken.npz"
TASK_COUNTS_RAW="$TASK_PROJECT/resources/token_freq_sample10.npz"
TASK_COUNTS_BALANCED="$TASK_PROJECT/resources/token_importance_langbalanced.npz"
TASK_RESULT_ROOT="$TASK_OUTPUT_BASE/hashed_zh_diagnostics_20260902_a01"
TASK_EXPORT_DIR="$TASK_OUTPUT_BASE/result_exports"
TASK_EXPORT="$TASK_EXPORT_DIR/hashed_zh_diagnostics_20260902_a01.tar.gz"
TASK_PARTIAL_BURN_LOG="$TASK_RESULT_ROOT/partial_burn_gpus_1_7.log"
TASK_FINAL_BURN_LOG="$TASK_RESULT_ROOT/final_all_gpu_burn.log"

TASK_LANGUAGES=(ar de en ru vi zh)
TASK_TRAJECTORY_STEPS=(1000 2500 5000 7500 10000)

gpu_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
}

gpu_uuids() {
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
        | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' \
        | awk 'NF' | sort -u
}

verify_direct_children() {
    local expected_parent="$1"
    shift
    local pid parent
    for pid in "$@"; do
        [[ "$pid" =~ ^[0-9]+$ && "$pid" != 1 ]]
        test -r "/proc/$pid/status"
        parent="$(awk '/^PPid:/ {print $2}' "/proc/$pid/status")"
        [[ "$parent" == "$expected_parent" ]] || {
            echo "REFUSE: GPU PID $pid has parent $parent, expected $expected_parent" >&2
            return 1
        }
    done
}

echo '=== immutable preflight before changing GPU state ==='
date -u
hostname
test -d "$TASK_PROJECT"
cd "$TASK_PROJECT"
echo "project=$TASK_PROJECT project_name=$TASK_PROJECT_NAME"
for TASK_PYTHON_CANDIDATE in "${TASK_PYTHON_CANDIDATES[@]}"; do
    if [[ -x "$TASK_PYTHON_CANDIDATE" ]]; then
        TASK_PYTHON="$TASK_PYTHON_CANDIDATE"
        break
    fi
done
if [[ -z "$TASK_PYTHON" ]]; then
    printf 'ERROR: eval Python not found; checked:\n' >&2
    printf '  %s\n' "${TASK_PYTHON_CANDIDATES[@]}" >&2
    exit 1
fi
echo "eval_python=$TASK_PYTHON"
test -x "$TASK_BURN_PYTHON"
test -s "$TASK_BURN_SCRIPT"
echo "$TASK_BURN_SHA  $TASK_BURN_SCRIPT" | sha256sum -c -
grep -F 'init_process_group(backend="nccl"' "$TASK_BURN_SCRIPT"
grep -F 'dist.all_reduce' "$TASK_BURN_SCRIPT"
test -s "$TASK_PROJECT/eval/ppl_bytoken.py"
test -s "$TASK_PROJECT/eval/ppl_bins.py"
test -s "$TASK_PROJECT/eval/hashed_zh_diagnostics.py"
sha256sum \
    "$TASK_PROJECT/eval/ppl_bytoken.py" \
    "$TASK_PROJECT/eval/ppl_bins.py" \
    "$TASK_PROJECT/eval/hashed_zh_diagnostics.py" \
    "$TASK_PROJECT/scripts/run_hashed_zh_diagnostics_b200.sh"
test -s "$TASK_COUNTS_RAW"
test -s "$TASK_COUNTS_BALANCED"
test -d "$TASK_DATA_DIR"
test -d "$TASK_TOKENIZER"
test -s "$TASK_DENSE_BYTOKEN"
test -s "$TASK_CONTROL_BYTOKEN"
test ! -e "$TASK_RESULT_ROOT"
test ! -e "$TASK_EXPORT"
test ! -e "$TASK_EXPORT.sha256"

for TASK_LANG in "${TASK_LANGUAGES[@]}"; do
    test -d "$TASK_DATA_DIR/$TASK_LANG"
done
for TASK_STEP in "${TASK_TRAJECTORY_STEPS[@]}"; do
    for TASK_MODEL_ROOT in "$TASK_HASHED_ROOT" "$TASK_CONTROL_ROOT"; do
        TASK_CKPT="$TASK_MODEL_ROOT/checkpoint-$TASK_STEP"
        test -d "$TASK_CKPT"
        test -s "$TASK_CKPT/config.json"
        test -s "$TASK_CKPT/model.safetensors"
        test -s "$TASK_CKPT/embedding.pt"
        test -s "$TASK_CKPT/trainer_state.json"
    done
done

"$TASK_PYTHON" - "$TASK_HASHED_ROOT" "$TASK_CONTROL_ROOT" \
    "$TASK_DENSE_BYTOKEN" "$TASK_CONTROL_BYTOKEN" \
    "$TASK_COUNTS_RAW" "$TASK_COUNTS_BALANCED" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

hashed_root, control_root = map(Path, sys.argv[1:3])
dense_bytoken, control_bytoken = map(Path, sys.argv[3:5])
raw_path, balanced_path = map(Path, sys.argv[5:7])
steps = (1000, 2500, 5000, 7500, 10000)
expected_languages = {"ar", "de", "en", "ru", "vi", "zh"}

for root, arm in ((hashed_root, "product_code"), (control_root, "groupreduce")):
    parent_config = json.loads((root / "train_config.json").read_text())
    comp = parent_config["compositional"]
    assert comp["arm"] == arm, (root, comp["arm"])
    assert comp["tie_output"] is True
    if arm == "product_code":
        assert comp["product_code_assignment"] == "hashed"
        assert comp["product_code_head_size"] == 2048
        assert comp["product_code_num_hashes"] == 4
        assert comp["product_code_num_buckets"] == 4096
        assert comp["product_code_seed"] == 0
    for step in steps:
        checkpoint = root / f"checkpoint-{step}"
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        assert state["global_step"] == step, (checkpoint, state["global_step"])
        config = json.loads((checkpoint / "config.json").read_text())
        assert config["vocab_size"] == 151936
        assert config["hidden_size"] == 1024
        assert config["tie_word_embeddings"] is False

for path in (dense_bytoken, control_bytoken):
    with np.load(path, allow_pickle=False) as data:
        languages = {key[:-4] for key in data.files if key.endswith("_nll")}
        assert languages == expected_languages, (path, languages)
        for language in languages:
            nll = data[f"{language}_nll"]
            count = data[f"{language}_cnt"]
            assert nll.shape == count.shape == (151936,)
            assert np.isfinite(nll).all() and np.all(nll >= 0)
            assert np.all(count >= 0) and int(count.sum()) > 0

with np.load(raw_path, allow_pickle=False) as data:
    assert set(data.files) == {
        "counts", "counts_ar", "counts_de", "counts_en",
        "counts_ru", "counts_vi", "counts_zh",
    }
    assert data["counts"].shape == (151936,)
    assert int(data["counts"].sum()) == 3501021467
with np.load(balanced_path, allow_pickle=False) as data:
    assert data.files == ["counts"]
    assert data["counts"].shape == (151936,)
    assert abs(float(data["counts"].sum()) - 1.0) < 1e-12
print("HASHED_DIAGNOSTIC_INPUTS_OK")
PY

"$TASK_PYTHON" - <<'PY'
import importlib.metadata
import torch

assert importlib.metadata.version("transformers") == "5.9.0"
assert importlib.metadata.version("datasets") == "4.8.5"
assert torch.cuda.is_available() and torch.cuda.device_count() == 8
names = [torch.cuda.get_device_name(index) for index in range(8)]
assert all("B200" in name for name in names), names
print(f"EVAL_ENV_OK torch={torch.__version__} gpus={names}")
PY

echo '=== verify current processes are exactly one eight-GPU runner burn ==='
mapfile -t TASK_OLD_GPU_PIDS < <(gpu_pids)
[[ "${#TASK_OLD_GPU_PIDS[@]}" -eq 8 ]] || {
    echo "REFUSE: expected 8 burn workers, found ${#TASK_OLD_GPU_PIDS[@]}" >&2
    exit 1
}
mapfile -t TASK_OLD_GPU_UUIDS < <(gpu_uuids)
[[ "${#TASK_OLD_GPU_UUIDS[@]}" -eq 8 ]] || {
    echo 'REFUSE: expected exactly one occupied context on every GPU' >&2
    exit 1
}
TASK_OLD_LAUNCHER=''
for TASK_PID in "${TASK_OLD_GPU_PIDS[@]}"; do
    TASK_PARENT="$(awk '/^PPid:/ {print $2}' "/proc/$TASK_PID/status")"
    if [[ -z "$TASK_OLD_LAUNCHER" ]]; then
        TASK_OLD_LAUNCHER="$TASK_PARENT"
    else
        [[ "$TASK_PARENT" == "$TASK_OLD_LAUNCHER" ]] || {
            echo 'REFUSE: GPU workers do not share one burn launcher' >&2
            exit 1
        }
    fi
done
[[ "$TASK_OLD_LAUNCHER" =~ ^[0-9]+$ && "$TASK_OLD_LAUNCHER" != 1 ]]
TASK_OLD_COMMAND="$(tr '\0' ' ' < "/proc/$TASK_OLD_LAUNCHER/cmdline")"
[[ "$TASK_OLD_COMMAND" == *"$TASK_BURN_SCRIPT"* ]] || {
    echo "REFUSE: GPU parent is not the runner burn: $TASK_OLD_COMMAND" >&2
    exit 1
}
verify_direct_children "$TASK_OLD_LAUNCHER" "${TASK_OLD_GPU_PIDS[@]}"
echo "VERIFIED_RUNNER_BURN launcher=$TASK_OLD_LAUNCHER workers=${TASK_OLD_GPU_PIDS[*]}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits

mkdir -p "$TASK_RESULT_ROOT" "$TASK_EXPORT_DIR"

echo '=== stop only the verified burn GPU workers ==='
kill -9 "${TASK_OLD_GPU_PIDS[@]}" 2>/dev/null || true
sleep 30
mapfile -t TASK_AFTER_KILL_PIDS < <(gpu_pids)
[[ "${#TASK_AFTER_KILL_PIDS[@]}" -eq 0 ]] || {
    echo "ERROR: GPU PIDs remain after burn stop: ${TASK_AFTER_KILL_PIDS[*]}" >&2
    exit 1
}
echo 'ALL_EIGHT_GPUS_FREE_30_SECONDS_AFTER_VERIFIED_BURN_STOP'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits

# From this point onward, handled failures still flow through the final burn handoff.
set +e
TASK_FAILED=0
declare -a TASK_PHASE1_PIDS=()
declare -a TASK_PHASE1_LABELS=()

launch_bytoken() {
    local gpu="$1"
    local language="$2"
    local output="$TASK_RESULT_ROOT/bytoken_shards/$language"
    mkdir -p "$output"
    echo "launch bytoken gpu=$gpu language=$language"
    env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        TOKENIZERS_PARALLELISM=false \
        HF_DATASETS_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        "$TASK_PYTHON" "$TASK_PROJECT/eval/ppl_bytoken.py" \
          --checkpoint "$TASK_HASHED_CKPT" \
          --eval-dir "$TASK_DATA_DIR" \
          --tokenizer-name "$TASK_TOKENIZER" \
          --device cuda --bf16 --langs "$language" \
          --output-dir "$output" \
          >"$output/run.log" 2>&1 &
    TASK_PHASE1_PIDS+=("$!")
    TASK_PHASE1_LABELS+=("bytoken:$language:gpu$gpu")
}

trajectory_worker() {
    local gpu="$1"
    shift
    local model_root name step checkpoint output
    while [[ "$#" -gt 0 ]]; do
        model_root="$1"
        name="$2"
        step="$3"
        shift 3
        checkpoint="$model_root/checkpoint-$step"
        output="$TASK_RESULT_ROOT/trajectory/$name/checkpoint-$step"
        mkdir -p "$output"
        echo "trajectory gpu=$gpu model=$name step=$step"
        env \
            CUDA_VISIBLE_DEVICES="$gpu" \
            TOKENIZERS_PARALLELISM=false \
            HF_DATASETS_OFFLINE=1 \
            TRANSFORMERS_OFFLINE=1 \
            "$TASK_PYTHON" "$TASK_PROJECT/eval/eval_checkpoint.py" \
              --checkpoint "$checkpoint" \
              --eval-dir "$TASK_DATA_DIR" \
              --tokenizer-name "$TASK_TOKENIZER" \
              --device cuda --bf16 --ppl-only --langs en zh \
              --output-dir "$output" \
              >"$output/run.log" 2>&1 || return 1
    done
}

for TASK_INDEX in "${!TASK_LANGUAGES[@]}"; do
    launch_bytoken "$TASK_INDEX" "${TASK_LANGUAGES[$TASK_INDEX]}"
done

trajectory_worker 6 \
    "$TASK_HASHED_ROOT" hashed 1000 \
    "$TASK_HASHED_ROOT" hashed 5000 \
    "$TASK_HASHED_ROOT" hashed 10000 \
    "$TASK_CONTROL_ROOT" control 2500 \
    "$TASK_CONTROL_ROOT" control 7500 \
    >"$TASK_RESULT_ROOT/trajectory_gpu6.log" 2>&1 &
TASK_PHASE1_PIDS+=("$!")
TASK_PHASE1_LABELS+=("trajectory:gpu6")

trajectory_worker 7 \
    "$TASK_HASHED_ROOT" hashed 2500 \
    "$TASK_HASHED_ROOT" hashed 7500 \
    "$TASK_CONTROL_ROOT" control 1000 \
    "$TASK_CONTROL_ROOT" control 5000 \
    "$TASK_CONTROL_ROOT" control 10000 \
    >"$TASK_RESULT_ROOT/trajectory_gpu7.log" 2>&1 &
TASK_PHASE1_PIDS+=("$!")
TASK_PHASE1_LABELS+=("trajectory:gpu7")

TASK_PHASE1_GPU_READY=0
for TASK_POLL in $(seq 1 12); do
    mapfile -t TASK_PHASE1_GPU_PIDS < <(gpu_pids)
    mapfile -t TASK_PHASE1_GPU_UUIDS < <(gpu_uuids)
    if [[ "${#TASK_PHASE1_GPU_PIDS[@]}" -eq 8 \
        && "${#TASK_PHASE1_GPU_UUIDS[@]}" -eq 8 ]]; then
        TASK_PHASE1_GPU_READY=1
        break
    fi
    sleep 5
done
for TASK_INDEX in "${!TASK_PHASE1_PIDS[@]}"; do
    if kill -0 "${TASK_PHASE1_PIDS[$TASK_INDEX]}" 2>/dev/null; then
        echo "PHASE1_STARTED ${TASK_PHASE1_LABELS[$TASK_INDEX]} pid=${TASK_PHASE1_PIDS[$TASK_INDEX]}"
    else
        echo "PHASE1_EARLY_EXIT ${TASK_PHASE1_LABELS[$TASK_INDEX]}" >&2
        TASK_FAILED=1
    fi
done
if [[ "$TASK_PHASE1_GPU_READY" -ne 1 ]]; then
    echo "PHASE1_GPU_PLACEMENT_FAILED pids=${#TASK_PHASE1_GPU_PIDS[@]} unique_gpus=${#TASK_PHASE1_GPU_UUIDS[@]}" >&2
    TASK_FAILED=1
else
    echo "PHASE1_ALL_EIGHT_GPUS_VERIFIED workers=${TASK_PHASE1_GPU_PIDS[*]}"
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits

for TASK_INDEX in "${!TASK_PHASE1_PIDS[@]}"; do
    if wait "${TASK_PHASE1_PIDS[$TASK_INDEX]}"; then
        echo "PHASE1_OK ${TASK_PHASE1_LABELS[$TASK_INDEX]}"
    else
        TASK_RC=$?
        echo "PHASE1_FAILED rc=$TASK_RC ${TASK_PHASE1_LABELS[$TASK_INDEX]}" >&2
        TASK_FAILED=1
    fi
done

for TASK_LANG in "${TASK_LANGUAGES[@]}"; do
    echo "=== bytoken tail: $TASK_LANG ==="
    tail -25 "$TASK_RESULT_ROOT/bytoken_shards/$TASK_LANG/run.log" 2>/dev/null || true
done
echo '=== trajectory worker tails ==='
tail -40 "$TASK_RESULT_ROOT/trajectory_gpu6.log" 2>/dev/null || true
tail -40 "$TASK_RESULT_ROOT/trajectory_gpu7.log" 2>/dev/null || true

sleep 30
mapfile -t TASK_POST_PHASE1_PIDS < <(gpu_pids)
if [[ "${#TASK_POST_PHASE1_PIDS[@]}" -ne 0 ]]; then
    echo "ERROR: GPU PIDs remain after phase 1: ${TASK_POST_PHASE1_PIDS[*]}" >&2
    TASK_FAILED=1
fi

TASK_HASHED_MERGED="$TASK_RESULT_ROOT/merged/hashed_eval_ppl_bytoken.npz"
if [[ "$TASK_FAILED" -eq 0 ]]; then
    echo '=== merge Hashed by-token shards and produce the all-language raw report ==='
    TASK_HASHED_RUN_ARGS=()
    for TASK_LANG in "${TASK_LANGUAGES[@]}"; do
        TASK_HASHED_RUN_ARGS+=(
            --run "hashed=$TASK_RESULT_ROOT/bytoken_shards/$TASK_LANG/eval_ppl_bytoken.npz"
        )
    done
    "$TASK_PYTHON" "$TASK_PROJECT/eval/ppl_bins.py" \
        --counts "$TASK_COUNTS_RAW" \
        "${TASK_HASHED_RUN_ARGS[@]}" \
        --write-merged-dir "$TASK_RESULT_ROOT/merged" \
        --output "$TASK_RESULT_ROOT/hashed_raw_all_languages.json" \
        >"$TASK_RESULT_ROOT/hashed_raw_all_languages.md" 2>&1
    TASK_RC=$?
    if [[ "$TASK_RC" -ne 0 ]]; then
        echo "HASHED_MERGE_FAILED rc=$TASK_RC" >&2
        TASK_FAILED=1
    fi
fi

run_bins() {
    local counts="$1"
    local populations="$2"
    local reference="$3"
    local output="$4"
    shift 4
    "$TASK_PYTHON" "$TASK_PROJECT/eval/ppl_bins.py" \
        --counts "$counts" \
        --populations "$populations" \
        --reference "$reference" \
        --run "dense=$TASK_DENSE_BYTOKEN" \
        --run "control=$TASK_CONTROL_BYTOKEN" \
        --run "hashed=$TASK_HASHED_MERGED" \
        --langs "$@" \
        --output "$output" \
        >"${output%.json}.md" 2>&1
}

if [[ "$TASK_FAILED" -eq 0 ]]; then
    echo '=== Checks 1 and 2: language-balanced and raw-frequency binning ==='
    run_bins "$TASK_COUNTS_BALANCED" 2048,149888 dense \
        "$TASK_RESULT_ROOT/check1_headtail_zh.json" zh || TASK_FAILED=1
    run_bins "$TASK_COUNTS_BALANCED" 2048,149888 dense \
        "$TASK_RESULT_ROOT/check1_headtail_nonzh.json" en ar de ru vi || TASK_FAILED=1
    run_bins "$TASK_COUNTS_BALANCED" 2048,6144,24576,119168 control \
        "$TASK_RESULT_ROOT/check2_balanced_tiers_zh.json" zh || TASK_FAILED=1
    run_bins "$TASK_COUNTS_BALANCED" 2048,6144,24576,119168 control \
        "$TASK_RESULT_ROOT/check2_balanced_tiers_nonzh.json" en ar de ru vi || TASK_FAILED=1
    run_bins "$TASK_COUNTS_RAW" 2048,6144,24576,119168 control \
        "$TASK_RESULT_ROOT/check2_raw_tiers_zh.json" zh || TASK_FAILED=1
    run_bins "$TASK_COUNTS_RAW" 2048,6144,24576,119168 control \
        "$TASK_RESULT_ROOT/check2_raw_tiers_nonzh.json" en ar de ru vi || TASK_FAILED=1
fi

if [[ "$TASK_FAILED" -eq 0 ]]; then
    echo '=== validate Checks 1, 2, and 6 outputs ==='
    "$TASK_PYTHON" - "$TASK_RESULT_ROOT" "$TASK_HASHED_CKPT/eval_ppl.json" \
        "$TASK_CONTROL_ROOT/checkpoint-10000/eval_ppl.json" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
hashed_reference = json.loads(Path(sys.argv[2]).read_text())
control_reference = json.loads(Path(sys.argv[3]).read_text())
languages = {"ar", "de", "en", "ru", "vi", "zh"}

summaries = {}
for language in languages:
    path = root / "bytoken_shards" / language / "eval_ppl_bytoken_summary.json"
    report = json.loads(path.read_text())
    assert set(report) == {language}
    row = report[language]
    assert row["max_chunk_gap"] <= 1e-3
    expected = hashed_reference[language]
    assert row["num_tokens"] == expected["num_tokens"]
    assert abs(row["loss"] - expected["loss"]) <= 1e-3
    assert abs(row["perplexity"] - expected["perplexity"]) / expected["perplexity"] <= 1e-3
    summaries[language] = row
assert set(summaries) == languages

bin_files = {
    "check1_headtail_zh.json": [2048, 149888],
    "check1_headtail_nonzh.json": [2048, 149888],
    "check2_balanced_tiers_zh.json": [2048, 6144, 24576, 119168],
    "check2_balanced_tiers_nonzh.json": [2048, 6144, 24576, 119168],
    "check2_raw_tiers_zh.json": [2048, 6144, 24576, 119168],
    "check2_raw_tiers_nonzh.json": [2048, 6144, 24576, 119168],
}
for filename, populations in bin_files.items():
    report = json.loads((root / filename).read_text())
    assert report["level_sets"]["type_count"] == populations
    section = report["level_sets"]
    assert set(section["ppl"]) == {"dense", "control", "hashed"}
    assert "mean_per_language_ppl" in section
    assert "per_language_gap_vs_reference" in section
    assert "mean_per_language_gap_vs_reference" in section
    for values in section["ppl"].values():
        assert all(value is not None and math.isfinite(value) for value in values)

curve = {"hashed": {}, "control": {}}
for model in curve:
    for step in (1000, 2500, 5000, 7500, 10000):
        path = root / "trajectory" / model / f"checkpoint-{step}" / "eval_ppl.json"
        report = json.loads(path.read_text())
        assert set(report) == {"en", "zh"}
        for language, row in report.items():
            assert row["num_tokens"] > 0
            assert all(math.isfinite(float(row[key])) for key in ("loss", "perplexity"))
        curve[model][str(step)] = report

for language in ("en", "zh"):
    assert abs(curve["hashed"]["10000"][language]["loss"] - hashed_reference[language]["loss"]) <= 1e-3
    assert abs(curve["control"]["10000"][language]["loss"] - control_reference[language]["loss"]) <= 1e-3
(root / "check6_trajectory_en_zh.json").write_text(
    json.dumps(curve, indent=2, allow_nan=False) + "\n"
)
print("CHECKS_1_2_6_VALIDATED")
PY
    [[ "$?" -eq 0 ]] || TASK_FAILED=1
fi

TASK_PARTIAL_LAUNCHER=''
if [[ "$TASK_FAILED" -eq 0 ]]; then
    echo '=== run a verified partial runner burn on GPUs 1-7 ==='
    nohup env \
        CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
        MASTER_ADDR=127.0.0.1 \
        MASTER_PORT=29537 \
        "$TASK_BURN_PYTHON" -u "$TASK_BURN_SCRIPT" \
        >"$TASK_PARTIAL_BURN_LOG" 2>&1 &
    TASK_PARTIAL_LAUNCHER=$!
    sleep 20
    mapfile -t TASK_PARTIAL_PIDS < <(
        nvidia-smi -i 1,2,3,4,5,6,7 \
            --query-compute-apps=pid --format=csv,noheader,nounits \
            | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
    )
    mapfile -t TASK_PARTIAL_UUIDS < <(
        nvidia-smi -i 1,2,3,4,5,6,7 \
            --query-compute-apps=gpu_uuid --format=csv,noheader \
            | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' \
            | awk 'NF' | sort -u
    )
    if [[ "${#TASK_PARTIAL_PIDS[@]}" -ne 7 ]] \
        || [[ "${#TASK_PARTIAL_UUIDS[@]}" -ne 7 ]] \
        || ! verify_direct_children "$TASK_PARTIAL_LAUNCHER" "${TASK_PARTIAL_PIDS[@]}"; then
        echo 'PARTIAL_BURN_START_FAILED' >&2
        TASK_FAILED=1
    else
        echo "PARTIAL_BURN_VERIFIED launcher=$TASK_PARTIAL_LAUNCHER workers=${TASK_PARTIAL_PIDS[*]}"
    fi
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits
fi

if [[ "$TASK_FAILED" -eq 0 ]]; then
    echo '=== Checks 3, 4, and 5 on GPU 0 ==='
    env \
        CUDA_VISIBLE_DEVICES=0 \
        TOKENIZERS_PARALLELISM=false \
        HF_DATASETS_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        "$TASK_PYTHON" "$TASK_PROJECT/eval/hashed_zh_diagnostics.py" \
          --checkpoint "$TASK_HASHED_CKPT" \
          --eval-dir "$TASK_DATA_DIR" \
          --tokenizer-name "$TASK_TOKENIZER" \
          --langs zh de \
          --device cuda --bf16 \
          --language-counts "$TASK_COUNTS_RAW" \
          --importance "$TASK_COUNTS_BALANCED" \
          --bytoken "$TASK_HASHED_MERGED" \
          --output "$TASK_RESULT_ROOT/checks_3_4_5.json" \
          >"$TASK_RESULT_ROOT/checks_3_4_5.log" 2>&1
    TASK_RC=$?
    if [[ "$TASK_RC" -ne 0 ]]; then
        echo "CHECKS_3_4_5_FAILED rc=$TASK_RC" >&2
        TASK_FAILED=1
    fi
fi

echo '=== stop only verified partial-burn GPU workers, if present ==='
mapfile -t TASK_PARTIAL_REMAINING < <(
    nvidia-smi -i 1,2,3,4,5,6,7 \
        --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
if [[ "${#TASK_PARTIAL_REMAINING[@]}" -gt 0 ]]; then
    if [[ -z "$TASK_PARTIAL_LAUNCHER" ]] \
        || ! verify_direct_children "$TASK_PARTIAL_LAUNCHER" "${TASK_PARTIAL_REMAINING[@]}"; then
        echo 'REFUSE: unexpected process on GPUs 1-7 during partial-burn cleanup' >&2
        exit 1
    fi
    kill -9 "${TASK_PARTIAL_REMAINING[@]}" 2>/dev/null || true
fi
sleep 30
mapfile -t TASK_POST_DIAGNOSTIC_PIDS < <(gpu_pids)
if [[ "${#TASK_POST_DIAGNOSTIC_PIDS[@]}" -ne 0 ]]; then
    echo "ERROR: GPU PIDs remain after diagnostics: ${TASK_POST_DIAGNOSTIC_PIDS[*]}" >&2
    exit 1
fi
echo 'ALL_EIGHT_GPUS_FREE_AFTER_DIAGNOSTICS'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits

if [[ "$TASK_FAILED" -eq 0 ]]; then
    echo '=== validate Checks 3, 4, and 5 report ==='
    "$TASK_PYTHON" - "$TASK_RESULT_ROOT/checks_3_4_5.json" <<'PY'
import json
import math
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
metadata = report["metadata"]
assert metadata["arm"] == "product_code"
assert metadata["assignment"] == "hashed"
assert metadata["product_code_seed"] == 0
assert metadata["tie_output"] is True
assert metadata["head_size"] == 2048
assert metadata["num_hashes"] == 4
assert metadata["num_buckets"] == 4096
assert metadata["max_positions"] == 0
assert set(report["prediction_leakage"]) == {"zh", "de"}
for language, row in report["prediction_leakage"].items():
    assert row["num_tail_positions"] > 0
    assert row["num_all_positions_seen"] >= row["num_tail_positions"]
    assert row["max_loss_verification_gap"] <= 1e-3
    assert row["truncated_at_max_positions"] is False
    for key, value in row.items():
        if isinstance(value, float):
            assert math.isfinite(value), (language, key, value)
state = report["state"]
assert set(state) == {
    "language_metadata", "row_norms", "cosine_similarity",
    "pair_sampling", "gates",
}
sampling = state["pair_sampling"]
assert sampling["requested_per_category"] == 20000
assert set(sampling["share_1plus_realized"]) == {"zh_zh", "zh_other", "other_other"}
assert all(value == 20000 for value in sampling["share_1plus_realized"].values())
assert all(value == 20000 for value in sampling["random_realized"].values())
assert set(state["gates"]["correlations"]["per_language_bytoken"]) == {
    "ar", "de", "en", "ru", "vi", "zh",
}
print("CHECKS_3_4_5_VALIDATED")
PY
    [[ "$?" -eq 0 ]] || TASK_FAILED=1
fi

echo '=== package available diagnostic results ==='
find "$TASK_RESULT_ROOT" -type f -printf '%P\n' | LC_ALL=C sort \
    >"$TASK_RESULT_ROOT/files.txt"
tar -C "$TASK_OUTPUT_BASE" -czf "$TASK_EXPORT" "$(basename "$TASK_RESULT_ROOT")"
sha256sum "$TASK_EXPORT" >"$TASK_EXPORT.sha256"
TASK_EXPORT_BYTES="$(stat -c %s "$TASK_EXPORT")"
echo "diagnostic_export_bytes=$TASK_EXPORT_BYTES"
sha256sum "$TASK_EXPORT"
if [[ "$TASK_EXPORT_BYTES" -ge 25000000 ]]; then
    echo 'WARNING: result archive exceeds the runner single-file pull limit' >&2
    TASK_FAILED=1
fi

echo '=== restore the exact runner burn on all eight GPUs ==='
nohup env \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=29547 \
    "$TASK_BURN_PYTHON" -u "$TASK_BURN_SCRIPT" \
    >"$TASK_FINAL_BURN_LOG" 2>&1 &
TASK_FINAL_BURN_LAUNCHER=$!
sleep 20
mapfile -t TASK_FINAL_BURN_PIDS < <(gpu_pids)
[[ "${#TASK_FINAL_BURN_PIDS[@]}" -eq 8 ]] || {
    echo "ERROR: final burn has ${#TASK_FINAL_BURN_PIDS[@]} workers, expected 8" >&2
    cat "$TASK_FINAL_BURN_LOG" >&2
    exit 1
}
verify_direct_children "$TASK_FINAL_BURN_LAUNCHER" "${TASK_FINAL_BURN_PIDS[@]}" || exit 1
mapfile -t TASK_FINAL_BURN_UUIDS < <(gpu_uuids)
[[ "${#TASK_FINAL_BURN_UUIDS[@]}" -eq 8 ]] || exit 1
for TASK_SAMPLE in 1 2 3; do
    echo "final_burn_sample=$TASK_SAMPLE"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv,noheader,nounits
    sleep 3
done
echo "FINAL_EIGHT_GPU_RUNNER_BURN_VERIFIED launcher=$TASK_FINAL_BURN_LAUNCHER workers=${TASK_FINAL_BURN_PIDS[*]}"

if [[ "$TASK_FAILED" -eq 0 ]]; then
    echo 'HASHED_ZH_DIAGNOSTICS_CHECKS_1_THROUGH_6_COMPLETE'
else
    echo 'HASHED_ZH_DIAGNOSTICS_FAILED; ALL_GPU_BURN_RESTORED' >&2
fi
exit "$TASK_FAILED"
