#2 +a
#th2-pull-frequency-binned-ppl-a02-burn-handoff-20260831-a01
set -euo pipefail

TASK_PROJECT=/mnt/local/@PROJECT@
TASK_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_BURN_PYTHON=/usr/bin/python3
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_DATA_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/eval
TASK_TOKENIZER=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_COUNTS="$TASK_PROJECT/resources/token_freq_sample10.npz"
TASK_RESULT_ROOT="$TASK_OUTPUT_BASE/frequency_binned_ppl_four_10k_20260831"
TASK_EXPORT_DIR="$TASK_OUTPUT_BASE/result_exports"
TASK_EXPORT="$TASK_EXPORT_DIR/frequency_binned_ppl_four_10k_20260831.tar.gz"
TASK_BURN_SOURCE="$TASK_PROJECT/resources/llm_pretrain_burn.py"
TASK_BURN_TARGET=/tmp/llm_pretrain_burn.py
TASK_BURN_LOG=/tmp/llm_pretrain_burn_all_gpus.log
TASK_BURN_PID_FILE=/tmp/llm_pretrain_burn_launcher.pid
TASK_OLD_BURN_SHA=97d96734d14f1dba578208a929f822e2693770f5f320daae2fbcff87853260aa
TASK_NEW_BURN_SHA=2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174

TASK_MODELS=(
  dense_tied_baseline_b200
  dense_tied_baseline_b200_ddp_default
  groupreduce_matched_nested_tied_t4
  nested_ladder_tied_t4
)
TASK_NAMES=(
  dense_ddp_false
  dense_ddp_default
  groupreduce_t4
  nested_ladder_t4
)

gpu_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d;s/[[:space:]]//g' | sort -nu
}

echo '=== preflight without changing GPU state ==='
date -u
hostname
test -d "$TASK_PROJECT"
test -x "$TASK_PYTHON"
test -x "$TASK_BURN_PYTHON"
test -s "$TASK_PROJECT/eval/ppl_bytoken.py"
test -s "$TASK_PROJECT/eval/ppl_bins.py"
test -s "$TASK_COUNTS"
test -d "$TASK_DATA_DIR"
test -d "$TASK_TOKENIZER"
test ! -e "$TASK_RESULT_ROOT"
test ! -e "$TASK_EXPORT"
echo "$TASK_NEW_BURN_SHA  $TASK_BURN_SOURCE" | sha256sum -c -

for TASK_LANG in ar de en ru vi zh; do
  test -d "$TASK_DATA_DIR/$TASK_LANG"
done
for TASK_INDEX in "${!TASK_MODELS[@]}"; do
  TASK_CKPT="$TASK_OUTPUT_BASE/${TASK_MODELS[$TASK_INDEX]}/checkpoint-10000"
  test -d "$TASK_CKPT"
  test -s "$TASK_CKPT/config.json"
  test -s "$TASK_CKPT/trainer_state.json"
  test -s "$TASK_CKPT/eval_ppl.json"
  TASK_WEIGHT_FILE="$(find "$TASK_CKPT" -maxdepth 1 -type f -name '*.safetensors' -print -quit)"
  test -n "$TASK_WEIGHT_FILE"
  echo "checkpoint_ok name=${TASK_NAMES[$TASK_INDEX]} path=$TASK_CKPT"
done

"$TASK_PYTHON" - "$TASK_COUNTS" "$TASK_OUTPUT_BASE" "${TASK_MODELS[@]}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

counts_path = Path(sys.argv[1])
output_base = Path(sys.argv[2])
models = sys.argv[3:]
expected_arms = [None, None, "groupreduce", "nested_ladder"]
assert len(models) == len(expected_arms)
with np.load(counts_path, allow_pickle=False) as archive:
    assert "counts" in archive.files
    counts = archive["counts"]
assert counts.shape == (151936,) and counts.dtype == np.int64
assert np.all(counts >= 0) and int(counts.sum()) == 3501021467
for model, expected_arm in zip(models, expected_arms):
    checkpoint = output_base / model / "checkpoint-10000"
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    assert state["global_step"] == 10000, (model, state["global_step"])
    config = json.loads((checkpoint / "config.json").read_text())
    assert config["vocab_size"] == 151936
    assert config["hidden_size"] == 1024
    if expected_arm is None:
        # Native dense Qwen tying is represented directly by the HF flag.
        assert config["tie_word_embeddings"] is True, model
        assert not (checkpoint / "embedding.pt").exists(), model
    else:
        # Custom exact tying deliberately disables HF's native tying because
        # EmbeddingShim and the structured head do not share a dense Parameter.
        assert config["tie_word_embeddings"] is False, model
        assert (checkpoint / "embedding.pt").is_file(), model
        train_config_path = checkpoint / "train_config.json"
        if not train_config_path.is_file():
            train_config_path = checkpoint.parent / "train_config.json"
        train_config = json.loads(train_config_path.read_text())
        compositional = train_config["compositional"]
        assert compositional["arm"] == expected_arm, (model, compositional["arm"])
        assert compositional["tie_output"] is True, model
print("FOUR_CHECKPOINT_PREFLIGHT_OK")
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

mkdir -p "$TASK_RESULT_ROOT/shards" "$TASK_RESULT_ROOT/merged" "$TASK_EXPORT_DIR"

echo '=== verify current GPU processes are exactly the known burn ==='
test -s "$TASK_BURN_TARGET"
TASK_CURRENT_BURN_SHA="$(sha256sum "$TASK_BURN_TARGET" | awk '{print $1}')"
case "$TASK_CURRENT_BURN_SHA" in
  "$TASK_OLD_BURN_SHA"|"$TASK_NEW_BURN_SHA") ;;
  *) echo "REFUSE: unexpected current burn hash $TASK_CURRENT_BURN_SHA" >&2; exit 1 ;;
esac
mapfile -t TASK_OLD_GPU_PIDS < <(gpu_pids)
[[ "${#TASK_OLD_GPU_PIDS[@]}" -eq 8 ]] || {
  echo "REFUSE: expected 8 burn GPU workers, found ${#TASK_OLD_GPU_PIDS[@]}" >&2
  exit 1
}
mapfile -t TASK_GPU_UUIDS < <(
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
    | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sort -u
)
[[ "${#TASK_GPU_UUIDS[@]}" -eq 8 ]] || {
  echo "REFUSE: expected one occupied context on each of 8 GPUs" >&2
  exit 1
}

TASK_OLD_LAUNCHER=''
for TASK_PID in "${TASK_OLD_GPU_PIDS[@]}"; do
  [[ "$TASK_PID" =~ ^[0-9]+$ && "$TASK_PID" != 1 ]]
  TASK_PPID="$(awk '/^PPid:/ {print $2}' "/proc/$TASK_PID/status")"
  [[ "$TASK_PPID" =~ ^[0-9]+$ && "$TASK_PPID" != 1 ]]
  if [[ -z "$TASK_OLD_LAUNCHER" ]]; then
    TASK_OLD_LAUNCHER="$TASK_PPID"
  else
    [[ "$TASK_PPID" == "$TASK_OLD_LAUNCHER" ]] || {
      echo 'REFUSE: current GPU workers do not share one burn launcher' >&2
      exit 1
    }
  fi
  ps -o pid=,ppid=,stat=,etime=,cmd= -p "$TASK_PID"
done
TASK_OLD_LAUNCHER_CMD="$(tr '\0' ' ' < "/proc/$TASK_OLD_LAUNCHER/cmdline")"
[[ "$TASK_OLD_LAUNCHER_CMD" == *"$TASK_BURN_TARGET"* ]] || {
  echo "REFUSE: launcher is not the runner burn: $TASK_OLD_LAUNCHER_CMD" >&2
  exit 1
}
echo "VERIFIED_OLD_BURN launcher=$TASK_OLD_LAUNCHER hash=$TASK_CURRENT_BURN_SHA"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

echo '=== kill only verified GPU compute workers ==='
kill -9 "${TASK_OLD_GPU_PIDS[@]}" 2>/dev/null || true
sleep 30
mapfile -t TASK_AFTER_KILL_PIDS < <(gpu_pids)
[[ "${#TASK_AFTER_KILL_PIDS[@]}" -eq 0 ]] || {
  echo "ERROR: GPU compute PIDs remain: ${TASK_AFTER_KILL_PIDS[*]}" >&2
  nvidia-smi
  exit 1
}
echo 'ALL_EIGHT_GPUS_FREE_AFTER_30_SECONDS'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

echo '=== launch eight independent by-token shards ==='
declare -a TASK_EVAL_PIDS=()
declare -a TASK_EVAL_LABELS=()

launch_shard() {
  local gpu="$1"
  local model="$2"
  local name="$3"
  local shard="$4"
  shift 4
  local checkpoint="$TASK_OUTPUT_BASE/$model/checkpoint-10000"
  local output="$TASK_RESULT_ROOT/shards/$name/$shard"
  mkdir -p "$output"
  echo "launch gpu=$gpu run=$name shard=$shard languages=$*"
  env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    TOKENIZERS_PARALLELISM=false \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    "$TASK_PYTHON" "$TASK_PROJECT/eval/ppl_bytoken.py" \
      --checkpoint "$checkpoint" \
      --eval-dir "$TASK_DATA_DIR" \
      --tokenizer-name "$TASK_TOKENIZER" \
      --device cuda \
      --bf16 \
      --langs "$@" \
      --output-dir "$output" \
      >"$output/run.log" 2>&1 &
  TASK_EVAL_PIDS+=("$!")
  TASK_EVAL_LABELS+=("gpu${gpu}:${name}:${shard}")
}

launch_shard 0 "${TASK_MODELS[0]}" "${TASK_NAMES[0]}" shard_a ar de en
launch_shard 1 "${TASK_MODELS[0]}" "${TASK_NAMES[0]}" shard_b ru vi zh
launch_shard 2 "${TASK_MODELS[1]}" "${TASK_NAMES[1]}" shard_a ar de en
launch_shard 3 "${TASK_MODELS[1]}" "${TASK_NAMES[1]}" shard_b ru vi zh
launch_shard 4 "${TASK_MODELS[2]}" "${TASK_NAMES[2]}" shard_a ar de en
launch_shard 5 "${TASK_MODELS[2]}" "${TASK_NAMES[2]}" shard_b ru vi zh
launch_shard 6 "${TASK_MODELS[3]}" "${TASK_NAMES[3]}" shard_a ar de en
launch_shard 7 "${TASK_MODELS[3]}" "${TASK_NAMES[3]}" shard_b ru vi zh

sleep 15
TASK_EVAL_FAILED=0
for TASK_PID in "${TASK_EVAL_PIDS[@]}"; do
  if ! kill -0 "$TASK_PID" 2>/dev/null; then
    echo "SHARD_EXITED_DURING_STARTUP pid=$TASK_PID" >&2
    TASK_EVAL_FAILED=1
  fi
done
if [[ "$TASK_EVAL_FAILED" -eq 0 ]]; then
  echo 'ALL_EIGHT_BYTOKEN_PROCESSES_STARTED'
else
  echo 'ONE_OR_MORE_BYTOKEN_PROCESSES_EXITED_DURING_STARTUP' >&2
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
  --format=csv,noheader

for TASK_INDEX in "${!TASK_EVAL_PIDS[@]}"; do
  TASK_PID="${TASK_EVAL_PIDS[$TASK_INDEX]}"
  TASK_LABEL="${TASK_EVAL_LABELS[$TASK_INDEX]}"
  if wait "$TASK_PID"; then
    echo "SHARD_OK $TASK_LABEL"
  else
    TASK_RC=$?
    echo "SHARD_FAILED rc=$TASK_RC $TASK_LABEL" >&2
    TASK_EVAL_FAILED=1
  fi
done

for TASK_INDEX in "${!TASK_NAMES[@]}"; do
  for TASK_SHARD in shard_a shard_b; do
    TASK_SHARD_DIR="$TASK_RESULT_ROOT/shards/${TASK_NAMES[$TASK_INDEX]}/$TASK_SHARD"
    echo "=== tail ${TASK_NAMES[$TASK_INDEX]}/$TASK_SHARD ==="
    tail -30 "$TASK_SHARD_DIR/run.log" || true
  done
done

if [[ "$TASK_EVAL_FAILED" -eq 0 ]]; then
  echo '=== validate shards against existing full-PPL results ==='
  if ! "$TASK_PYTHON" - "$TASK_RESULT_ROOT" "$TASK_OUTPUT_BASE" \
      "${TASK_NAMES[@]}" -- "${TASK_MODELS[@]}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
output_base = Path(sys.argv[2])
separator = sys.argv.index("--")
names = sys.argv[3:separator]
models = sys.argv[separator + 1:]
assert len(names) == len(models) == 4
expected_languages = {"ar", "de", "en", "ru", "vi", "zh"}

for name, model in zip(names, models):
    summaries = {}
    for shard in ("shard_a", "shard_b"):
        directory = root / "shards" / name / shard
        npz_path = directory / "eval_ppl_bytoken.npz"
        summary_path = directory / "eval_ppl_bytoken_summary.json"
        assert npz_path.is_file() and summary_path.is_file(), (name, shard)
        summary = json.loads(summary_path.read_text())
        overlap = set(summaries).intersection(summary)
        assert not overlap, (name, shard, overlap)
        summaries.update(summary)
    assert set(summaries) == expected_languages, (name, sorted(summaries))
    reference = json.loads(
        (output_base / model / "checkpoint-10000" / "eval_ppl.json").read_text()
    )
    assert set(reference) == expected_languages
    for language in sorted(expected_languages):
        result = summaries[language]
        expected = reference[language]
        assert result["num_tokens"] == expected["num_tokens"]
        assert result["max_chunk_gap"] <= 1e-3, (name, language, result)
        assert abs(result["loss"] - expected["loss"]) <= 1e-3, (
            name, language, result["loss"], expected["loss"]
        )
        assert abs(result["perplexity"] - expected["perplexity"]) \
            / expected["perplexity"] <= 1e-3
        assert all(math.isfinite(float(value)) for value in result.values())
    print(f"SHARDS_VALIDATED run={name} languages=6")
print("ALL_EIGHT_SHARDS_MATCH_EXISTING_PPL")
PY
  then
    TASK_EVAL_FAILED=1
  fi
fi

if [[ "$TASK_EVAL_FAILED" -eq 0 ]]; then
  echo '=== aggregate level-set and mass-bin PPL on CPU ==='
  if ! "$TASK_PYTHON" "$TASK_PROJECT/eval/ppl_bins.py" \
      --counts "$TASK_COUNTS" \
      --run "dense_ddp_false=$TASK_RESULT_ROOT/shards/dense_ddp_false/shard_a/eval_ppl_bytoken.npz" \
      --run "dense_ddp_false=$TASK_RESULT_ROOT/shards/dense_ddp_false/shard_b/eval_ppl_bytoken.npz" \
      --run "dense_ddp_default=$TASK_RESULT_ROOT/shards/dense_ddp_default/shard_a/eval_ppl_bytoken.npz" \
      --run "dense_ddp_default=$TASK_RESULT_ROOT/shards/dense_ddp_default/shard_b/eval_ppl_bytoken.npz" \
      --run "groupreduce_t4=$TASK_RESULT_ROOT/shards/groupreduce_t4/shard_a/eval_ppl_bytoken.npz" \
      --run "groupreduce_t4=$TASK_RESULT_ROOT/shards/groupreduce_t4/shard_b/eval_ppl_bytoken.npz" \
      --run "nested_ladder_t4=$TASK_RESULT_ROOT/shards/nested_ladder_t4/shard_a/eval_ppl_bytoken.npz" \
      --run "nested_ladder_t4=$TASK_RESULT_ROOT/shards/nested_ladder_t4/shard_b/eval_ppl_bytoken.npz" \
      --reference dense_ddp_default \
      --write-merged-dir "$TASK_RESULT_ROOT/merged" \
      --output "$TASK_RESULT_ROOT/frequency_binned_ppl.json" \
      | tee "$TASK_RESULT_ROOT/frequency_binned_ppl.md"
  then
    TASK_EVAL_FAILED=1
  fi
fi

if [[ "$TASK_EVAL_FAILED" -eq 0 ]]; then
  if ! "$TASK_PYTHON" - "$TASK_RESULT_ROOT/frequency_binned_ppl.json" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text())
assert report["metadata"]["languages"] == ["ar", "de", "en", "ru", "vi", "zh"]
assert report["level_sets"]["type_count"] == [2048, 6144, 24576, 119168]
assert report["mass_bins"]["type_count"] == [19130, 43898, 88908]
expected_runs = {
    "dense_ddp_false", "dense_ddp_default", "groupreduce_t4", "nested_ladder_t4"
}
for section_name in ("level_sets", "mass_bins"):
    section = report[section_name]
    assert set(section["ppl"]) == expected_runs
    assert abs(sum(section["training_mass_share"]) - 1.0) < 1e-12
    assert abs(sum(section["eval_token_share"]) - 1.0) < 1e-12
    for values in section["ppl"].values():
        assert all(value is not None and math.isfinite(value) for value in values)
    for gap in section["gap_vs_reference"].values():
        assert abs(
            sum(gap["per_bin_contribution"]) - gap["total_mean_log_ppl_gap"]
        ) < 1e-12
for run in expected_runs:
    merged = path.parent / "merged" / f"{run}_eval_ppl_bytoken.npz"
    assert merged.is_file() and merged.stat().st_size > 0
print("FREQUENCY_BINNED_REPORT_VALIDATED")
PY
  then
    TASK_EVAL_FAILED=1
  fi
fi

if [[ "$TASK_EVAL_FAILED" -eq 0 ]]; then
  echo '=== package compact result bundle ==='
  find "$TASK_RESULT_ROOT" -type f -printf '%P\n' | LC_ALL=C sort \
    > "$TASK_RESULT_ROOT/files.txt"
  tar -C "$TASK_OUTPUT_BASE" -czf "$TASK_EXPORT" \
    "$(basename "$TASK_RESULT_ROOT")"
  sha256sum "$TASK_EXPORT" > "$TASK_EXPORT.sha256"
  sha256sum "$TASK_EXPORT"
  echo 'FREQUENCY_BINNED_PPL_COMPLETE_AND_PACKAGED'
else
  echo 'FREQUENCY_BINNED_PPL_FAILED; HANDOFF WILL STILL RESTORE CORRECT BURNS' >&2
fi

echo '=== wait 30 seconds and require all GPUs free ==='
sleep 30
mapfile -t TASK_POST_EVAL_PIDS < <(gpu_pids)
[[ "${#TASK_POST_EVAL_PIDS[@]}" -eq 0 ]] || {
  echo "ERROR: GPU PIDs remain after evaluation: ${TASK_POST_EVAL_PIDS[*]}" >&2
  nvidia-smi
  exit 1
}
echo 'ALL_EIGHT_GPUS_FREE_AFTER_FREQUENCY_BINNED_PPL'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

echo '=== install and start the newest correct all-GPU burn ==='
install -m 0644 "$TASK_BURN_SOURCE" "$TASK_BURN_TARGET"
echo "$TASK_NEW_BURN_SHA  $TASK_BURN_TARGET" | sha256sum -c -
rm -f "$TASK_BURN_LOG" "$TASK_BURN_PID_FILE"
nohup env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT=29631 \
  GPU_BURN_MEMORY_FRACTION=0.85 \
  GPU_BURN_MIN_FREE_GIB=8 \
  GPU_BURN_MATRIX_SIZE=8192 \
  GPU_BURN_COMM_TOTAL_MIB=1137 \
  GPU_BURN_COMM_BUCKET_MIB=25 \
  GPU_BURN_APPROX_STEP_SECONDS=0.75 \
  NCCL_DEBUG=WARN \
  "$TASK_BURN_PYTHON" -u "$TASK_BURN_TARGET" \
  >"$TASK_BURN_LOG" 2>&1 &
TASK_NEW_LAUNCHER=$!
printf '%s\n' "$TASK_NEW_LAUNCHER" > "$TASK_BURN_PID_FILE"
echo "new_burn_launcher=$TASK_NEW_LAUNCHER"

TASK_BURN_READY=0
for _ in $(seq 1 240); do
  if ! kill -0 "$TASK_NEW_LAUNCHER" 2>/dev/null; then
    echo 'ERROR: new burn launcher exited' >&2
    cat "$TASK_BURN_LOG" >&2
    exit 1
  fi
  TASK_READY_COUNT="$(grep -Fc 'gpu_burn_ready' "$TASK_BURN_LOG" 2>/dev/null || true)"
  mapfile -t TASK_NEW_GPU_PIDS < <(gpu_pids)
  if [[ "$TASK_READY_COUNT" -eq 8 && "${#TASK_NEW_GPU_PIDS[@]}" -eq 8 ]]; then
    TASK_BURN_READY=1
    break
  fi
  sleep 1
done
[[ "$TASK_BURN_READY" -eq 1 ]] || {
  echo 'ERROR: correct burn did not become ready on all 8 GPUs' >&2
  cat "$TASK_BURN_LOG" >&2
  exit 1
}

for _ in $(seq 1 60); do
  grep -Fq 'gpu_burn_progress' "$TASK_BURN_LOG" && break
  sleep 1
done
grep -Fq 'gpu_burn_progress' "$TASK_BURN_LOG"
[[ "$(grep -Fc 'world_size=8' "$TASK_BURN_LOG")" -eq 8 ]]
[[ "$(grep -Fc 'collective_probe_sum=36' "$TASK_BURN_LOG")" -eq 8 ]]
[[ "$(grep -Fc 'comm_total_mib=1137' "$TASK_BURN_LOG")" -eq 8 ]]
[[ "$(grep -Fc 'comm_bucket_mib=25' "$TASK_BURN_LOG")" -eq 8 ]]
[[ "$(grep -Fc 'approx_step_seconds=0.750' "$TASK_BURN_LOG")" -eq 8 ]]

"$TASK_BURN_PYTHON" - "$TASK_NEW_LAUNCHER" <<'PY'
import csv
import statistics
import subprocess
import sys
import time
from pathlib import Path

launcher = int(sys.argv[1])

def run(*args):
    return subprocess.check_output(args, text=True).strip()

def rows(output):
    return [
        [field.strip() for field in row]
        for row in csv.reader(output.splitlines())
        if row
    ]

def parent(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    raise RuntimeError(f"missing parent for {pid}")

def descendant(pid):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == launcher:
            return True
        seen.add(pid)
        pid = parent(pid)
    return False

gpu_rows = rows(run(
    "nvidia-smi", "--query-gpu=index,uuid,name,memory.total",
    "--format=csv,noheader,nounits"
))
assert len(gpu_rows) == 8
gpus = {}
totals = {}
for index_text, uuid, name, total_text in gpu_rows:
    index = int(index_text)
    assert index in range(8) and index not in gpus and "B200" in name
    gpus[index] = uuid
    totals[index] = float(total_text)
uuid_to_index = {uuid: index for index, uuid in gpus.items()}

app_rows = rows(run(
    "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
    "--format=csv,noheader,nounits"
))
assert len(app_rows) == 8, app_rows
seen = set()
for uuid, pid_text, memory_text in app_rows:
    assert uuid in uuid_to_index
    index = uuid_to_index[uuid]
    assert index not in seen
    seen.add(index)
    assert descendant(int(pid_text)), (index, pid_text, launcher)
assert seen == set(range(8))

samples = {index: [] for index in range(8)}
for sample_number in range(1, 6):
    sample_rows = rows(run(
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits"
    ))
    assert len(sample_rows) == 8
    for index_text, memory_text, utilization_text, power_text in sample_rows:
        index = int(index_text)
        values = (float(memory_text), float(utilization_text), float(power_text))
        samples[index].append(values)
    print(
        f"sample={sample_number} " + " | ".join(
            f"GPU{index} mem={samples[index][-1][0]:.0f}MiB "
            f"util={samples[index][-1][1]:.0f}% power={samples[index][-1][2]:.0f}W"
            for index in range(8)
        )
    )
    if sample_number < 5:
        time.sleep(2)

for index in range(8):
    memory_fractions = [sample[0] / totals[index] for sample in samples[index]]
    utilizations = [sample[1] for sample in samples[index]]
    assert min(memory_fractions) >= 0.82, (index, memory_fractions)
    assert max(memory_fractions) <= 0.88, (index, memory_fractions)
    assert max(utilizations) >= 90, (index, utilizations)
    assert statistics.mean(utilizations) >= 75, (index, utilizations)
print("CORRECT_BURN_ALL_EIGHT_GPUS_85_PERCENT_HBM")
print("CORRECT_BURN_NCCL_CONFIGURATION_AND_PROGRESS_VERIFIED")
PY

tail -80 "$TASK_BURN_LOG"
nvidia-smi
echo 'CORRECT_GPU_BURNS_ACTIVE_ON_ALL_EIGHT_GPUS'

if [[ "$TASK_EVAL_FAILED" -ne 0 ]]; then
  exit 1
fi
echo 'TH2 FREQUENCY-BINNED PPL FOUR CHECKPOINTS COMPLETE'
