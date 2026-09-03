#1 +300+a
#th2-train-ranklift-hashedv2-btmos-10k-20260903-a01
set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

read_gpu_pids() {
    local destination_name="$1"
    local output
    local -n destination="$destination_name"
    output="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" \
        || return 1
    mapfile -t destination < <(
        printf '%s\n' "$output" |
            sed 's/^[[:space:]]*//;s/[[:space:]]*$//;/^$/d' | sort -nu
    )
}

require_b200_node() {
    mapfile -t TASK_GPU_NAMES < <(
        nvidia-smi --query-gpu=name --format=csv,noheader |
            sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
    )
    [[ "${#TASK_GPU_NAMES[@]}" -eq 8 ]] \
        || die "expected 8 GPUs, found ${#TASK_GPU_NAMES[@]}"
    for TASK_GPU_INDEX in "${!TASK_GPU_NAMES[@]}"; do
        [[ "${TASK_GPU_NAMES[$TASK_GPU_INDEX]}" == *B200* ]] \
            || die "GPU $TASK_GPU_INDEX is not B200: ${TASK_GPU_NAMES[$TASK_GPU_INDEX]}"
    done
}

require_free_gpus() {
    local stage="$1"
    TASK_STAGE_GPU_PIDS=()
    read_gpu_pids TASK_STAGE_GPU_PIDS \
        || die "nvidia-smi process query failed $stage"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv,noheader
    [[ "${#TASK_STAGE_GPU_PIDS[@]}" -eq 0 ]] \
        || die "GPU compute processes remain $stage: ${TASK_STAGE_GPU_PIDS[*]}"
    echo "ALL 8 B200 GPUS FREE $stage"
}

start_burn_and_verify() {
    local burn_log="$1"
    test -s /tmp/llm_pretrain_burn.py || return 1
    TASK_PRE_BURN_PIDS=()
    read_gpu_pids TASK_PRE_BURN_PIDS || return 1
    [[ "${#TASK_PRE_BURN_PIDS[@]}" -eq 0 ]] || {
        echo "REFUSE BURN: GPU processes already exist: ${TASK_PRE_BURN_PIDS[*]}" >&2
        return 1
    }

    mkdir -p "$(dirname "$burn_log")"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        python3 /tmp/llm_pretrain_burn.py >>"$burn_log" 2>&1 &
    TASK_BURN_LAUNCHER_PID=$!
    echo "burn_launcher_pid=$TASK_BURN_LAUNCHER_PID"
    sleep 30

    TASK_BURN_GPU_PIDS=()
    read_gpu_pids TASK_BURN_GPU_PIDS || return 1
    [[ "${#TASK_BURN_GPU_PIDS[@]}" -eq 8 ]] || {
        echo "ERROR: expected 8 burn GPU workers, found ${#TASK_BURN_GPU_PIDS[@]}" >&2
        nvidia-smi
        return 1
    }
    for TASK_GPU_INDEX in 0 1 2 3 4 5 6 7; do
        TASK_ONE_GPU_PID_OUTPUT="$(
            nvidia-smi -i "$TASK_GPU_INDEX" \
                --query-compute-apps=pid --format=csv,noheader,nounits
        )" || return 1
        mapfile -t TASK_ONE_GPU_PIDS < <(
            printf '%s\n' "$TASK_ONE_GPU_PID_OUTPUT" |
                sed 's/^[[:space:]]*//;s/[[:space:]]*$//;/^$/d' | sort -nu
        )
        [[ "${#TASK_ONE_GPU_PIDS[@]}" -eq 1 ]] || {
            echo "ERROR: GPU $TASK_GPU_INDEX has ${#TASK_ONE_GPU_PIDS[@]} burn workers" >&2
            return 1
        }
        kill -0 "${TASK_ONE_GPU_PIDS[0]}" || return 1
        echo "gpu=$TASK_GPU_INDEX burn_worker_pid=${TASK_ONE_GPU_PIDS[0]}"
    done
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv,noheader
    echo 'ALL 8 RUNNER GPU BURNS ACTIVE'
}

restore_burn_on_exit() {
    local rc=$?
    trap - EXIT
    set +e
    if [[ "${TASK_BURN_RESTORED:-0}" -eq 0 ]]; then
        TASK_EXIT_GPU_PIDS=()
        if ! read_gpu_pids TASK_EXIT_GPU_PIDS; then
            echo 'ERROR: cannot query GPU processes during exit; burn not started' >&2
        elif [[ "${#TASK_EXIT_GPU_PIDS[@]}" -eq 0 ]]; then
            echo "=== restoring runner burn during exit rc=$rc ==="
            if start_burn_and_verify "$TASK_BURN_LOG"; then
                TASK_BURN_RESTORED=1
            else
                echo 'ERROR: could not restore runner burn during exit' >&2
            fi
        else
            echo "BURN NOT STARTED: GPU processes remain at exit: ${TASK_EXIT_GPU_PIDS[*]}" >&2
        fi
    fi
    exit "$rc"
}

TASK_PROJECT_DIR=/mnt/local/@PROJECT@
TASK_CONDA=/mnt/local/conda-py311/bin/conda
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_DATA_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/train
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/ranklift_hashedv2_btmos_10k_20260903_a01"
TASK_STATUS_DIR="$TASK_OUTPUT_BASE/status"
TASK_COMPLETION_FILE="$TASK_STATUS_DIR/ranklift_hashedv2_btmos_10k_20260903_a01.complete"
TASK_COMPLETION_TMP="$TASK_COMPLETION_FILE.tmp"
TASK_BURN_LOG="$TASK_OUTPUT_BASE/logs/gpu_burn_after_ranklift_hashedv2_btmos_20260903_a01.log"
TASK_BTMOS_SMOKE_DIR="$TASK_PROJECT_DIR/temp/btmos_pretrain_gate_20260903_a01"
TASK_ACCELERATE_SOURCE="$TASK_PROJECT_DIR/resources/accelerate_config.yaml"
TASK_ACCELERATE_TARGET=/mnt/local/.cache/huggingface/accelerate/default_config.yaml
TASK_EXPERIMENT_NAMES=(
    ranklift_tied_c124_m460
    product_code_quota_h6144
    btmos_k3_c256_lb
)
TASK_OUTPUT_DIRS=(
    "$TASK_OUTPUT_BASE/ranklift_tied_c124_m460"
    "$TASK_OUTPUT_BASE/product_code_quota_h6144"
    "$TASK_OUTPUT_BASE/btmos_k3_c256_lb"
)
TASK_LAUNCHERS=(
    scripts/train_ranklift_tied.sh
    scripts/train_product_code_quota_tied.sh
    scripts/train_btmos_tied.sh
)
TASK_BURN_RESTORED=0
trap restore_burn_on_exit EXIT

echo '=== identity and static preflight ==='
date -u
hostname
cd "$TASK_PROJECT_DIR"
test -x "$TASK_CONDA"
test -x "$TASK_PYTHON"
test -s "$TASK_MODEL_DIR/config.json"
test -s "$TASK_MODEL_DIR/tokenizer.json"
test -d "$TASK_DATA_DIR"
test -s "$TASK_ACCELERATE_SOURCE"
test -s /tmp/llm_pretrain_burn.py
for TASK_LAUNCHER in "${TASK_LAUNCHERS[@]}"; do
    test -s "$TASK_LAUNCHER"
    grep -Eq '^[[:space:]]*--bf16([[:space:]]|$)' "$TASK_LAUNCHER" \
        || die "missing bf16 in $TASK_LAUNCHER"
    grep -Eq '^[[:space:]]*--ddp_find_unused_parameters[[:space:]]+false([[:space:]]|$)' \
        "$TASK_LAUNCHER" || die "missing DDP false in $TASK_LAUNCHER"
    grep -Eq '^[[:space:]]*--per_device_train_batch_size[[:space:]]+16([[:space:]]|$)' \
        "$TASK_LAUNCHER" || die "wrong per-device batch in $TASK_LAUNCHER"
    grep -Eq '^[[:space:]]*--gradient_accumulation_steps[[:space:]]+4([[:space:]]|$)' \
        "$TASK_LAUNCHER" || die "wrong accumulation in $TASK_LAUNCHER"
    grep -Eq '^[[:space:]]*--save_steps[[:space:]]+250([[:space:]]|$)' \
        "$TASK_LAUNCHER" || die "wrong save interval in $TASK_LAUNCHER"
    grep -Eq '^[[:space:]]*--tie_output([[:space:]]|$)' "$TASK_LAUNCHER" \
        || die "missing exact tied output in $TASK_LAUNCHER"
    if grep -Eq '^[[:space:]]*--max_steps([[:space:]]|$)' "$TASK_LAUNCHER"; then
        die "launcher must retain the one-epoch schedule: $TASK_LAUNCHER"
    fi
done
echo '39b15eab8cf213d563dcf5137bb982e836bb8e3beba8e7def8dcddf21fe43594  resources/token_importance_langbalanced.npz' \
    | sha256sum -c -
echo 'f43d19925f5add96c56913eccf57f3989d6cd52e69da761d879e22f901010ea5  resources/token_importance_quota.npz' \
    | sha256sum -c -
echo '923db7f20a2df3d051180f67f9bea1f30c84c804651e313fa9961a9fd17a57e5  resources/accelerate_config.yaml' \
    | sha256sum -c -
for TASK_LANG in en vi zh ru de ar; do
    test -d "$TASK_DATA_DIR/$TASK_LANG" \
        || die "missing training language directory: $TASK_LANG"
    find "$TASK_DATA_DIR/$TASK_LANG" -type f -name '*.arrow' \
        ! -name 'cache-*' -print -quit | grep -q . \
        || die "no source Arrow file for $TASK_LANG"
done
for TASK_OUTPUT_DIR in "${TASK_OUTPUT_DIRS[@]}"; do
    [[ ! -e "$TASK_OUTPUT_DIR" ]] \
        || die "refusing non-fresh experiment output: $TASK_OUTPUT_DIR"
done
[[ ! -e "$TASK_LOG_DIR" ]] || die "refusing existing log directory: $TASK_LOG_DIR"
[[ ! -e "$TASK_BTMOS_SMOKE_DIR" ]] \
    || die "refusing existing BT-MoS smoke directory: $TASK_BTMOS_SMOKE_DIR"
[[ ! -e "$TASK_COMPLETION_FILE" && ! -e "$TASK_COMPLETION_TMP" ]] \
    || die "refusing existing completion marker: $TASK_COMPLETION_FILE"

echo '=== activate and verify sparse_emb ==='
eval "$("$TASK_CONDA" shell.bash hook)"
conda activate sparse_emb
[[ "${CONDA_DEFAULT_ENV:-}" == sparse_emb ]] || die 'failed to activate sparse_emb'
[[ "$(command -v python3.11)" == "$TASK_PYTHON" ]] \
    || die "wrong Python after activation: $(command -v python3.11)"
"$TASK_PYTHON" - "$TASK_MODEL_DIR/config.json" <<'PY'
import json
import sys

import accelerate
import datasets
import torch
import transformers

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
assert config["model_type"] == "qwen3"
assert config["hidden_size"] == 1024
assert config["vocab_size"] == 151936
assert config["tie_word_embeddings"] is True
assert torch.cuda.is_available() and torch.cuda.device_count() == 8
assert all("B200" in torch.cuda.get_device_name(i) for i in range(8))
print(
    "TRAINING_RUNTIME_OK",
    f"torch={torch.__version__}",
    f"transformers={transformers.__version__}",
    f"datasets={datasets.__version__}",
    f"accelerate={accelerate.__version__}",
)
PY
require_b200_node

echo '=== prove no real project GPU workload precedes burn reclamation ==='
if pgrep -af '[r]un_experiments.py|[t]rain_compositional.py|[t]rain.py|[e]val_parallel.py|[f]inetune_parallel.py|[p]repare_data.py'; then
    die 'training, evaluation, finetuning, or data-preparation process exists before burn reclamation'
fi
TASK_BURN_PIDS=()
read_gpu_pids TASK_BURN_PIDS \
    || die 'failed to query current GPU compute PIDs'
[[ "${#TASK_BURN_PIDS[@]}" -eq 8 ]] \
    || die "expected exactly 8 current burn GPU workers, found ${#TASK_BURN_PIDS[@]}"
for TASK_PID in "${TASK_BURN_PIDS[@]}"; do
    [[ "$TASK_PID" =~ ^[0-9]+$ ]] || die "invalid GPU PID: $TASK_PID"
    [[ "$TASK_PID" -ne 1 ]] || die 'refusing to signal PID 1'
    TASK_PPID="$(awk '/^PPid:/ {print $2}' "/proc/$TASK_PID/status")"
    TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
    TASK_PARENT_CMDLINE='unavailable'
    if [[ -r "/proc/$TASK_PPID/cmdline" ]]; then
        TASK_PARENT_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PPID/cmdline")"
    fi
    echo "verified_gpu_worker=$TASK_PID ppid=$TASK_PPID cmdline=$TASK_CMDLINE parent=$TASK_PARENT_CMDLINE"
done

echo '=== stop the eight verified GPU burn workers only ==='
kill -9 "${TASK_BURN_PIDS[@]}" 2>/dev/null || true
echo '=== sleep 30 seconds after burn stop ==='
sleep 30
require_free_gpus '30 SECONDS AFTER BURN STOP'

echo '=== copy and verify exact 8-GPU bf16 Accelerate config ==='
mkdir -p "$(dirname "$TASK_ACCELERATE_TARGET")"
cp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
cmp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
grep -Fxq 'distributed_type: MULTI_GPU' "$TASK_ACCELERATE_TARGET"
grep -Fxq 'mixed_precision: bf16' "$TASK_ACCELERATE_TARGET"
grep -Fxq 'num_processes: 8' "$TASK_ACCELERATE_TARGET"
echo '=== sleep 30 seconds after Accelerate config copy ==='
sleep 30
require_free_gpus '30 SECONDS AFTER ACCELERATE CONFIG COPY'

export SPARSE_EMB_PYTHON="$TASK_PYTHON"
export SPARSE_EMB_MODEL_DIR="$TASK_MODEL_DIR"
export SPARSE_EMB_DATA_DIR="$TASK_DATA_DIR"
export SPARSE_EMB_OUTPUT_BASE="$TASK_OUTPUT_BASE"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo '=== run final 8-GPU BF16 BT-MoS train/resume/load smoke ==='
"$TASK_PYTHON" -u scripts/smoke_btmos_e2e.py \
    --scratch "$TASK_BTMOS_SMOKE_DIR" \
    --gpus 8 \
    --tokenizer "$TASK_MODEL_DIR" \
    --port 29603
"$TASK_PYTHON" - "$TASK_BTMOS_SMOKE_DIR/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
assert summary["overall"] == "PASS", summary
assert summary["gpus"] == 8, summary
assert summary["train_fresh"]["exit"] == 0, summary
assert summary["train_resume"]["exit"] == 0, summary
assert summary["ppl_bytoken"]["exit"] == 0, summary
assert summary["eval_checkpoint"]["exit"] == 0, summary
print("BTMOS_8GPU_BF16_E2E_GATE_PASS")
PY
require_free_gpus 'AFTER BT-MOS 8-GPU BF16 E2E GATE'

echo '=== resolve exact experiments by name ==='
mapfile -t TASK_EXPERIMENT_INDICES < <(
    "$TASK_PYTHON" - <<'PY'
from run_experiments import EXPERIMENT_COMMANDS

expected = (
    ("ranklift_tied_c124_m460", "bash scripts/train_ranklift_tied.sh"),
    (
        "product_code_quota_h6144",
        "PRODUCT_CODE_QUOTA_HEAD_SIZE=6144 "
        "bash scripts/train_product_code_quota_tied.sh",
    ),
    ("btmos_k3_c256_lb", "bash scripts/train_btmos_tied.sh"),
)
for name, command in expected:
    matches = [
        (index, experiment)
        for index, experiment in enumerate(EXPERIMENT_COMMANDS)
        if experiment.get("name") == name
    ]
    assert len(matches) == 1, (name, matches)
    index, experiment = matches[0]
    assert experiment["cmd"] == command, experiment
    assert experiment.get("require_fresh_output") is True, experiment
    print(index)
PY
)
[[ "${#TASK_EXPERIMENT_INDICES[@]}" -eq 3 ]] \
    || die 'failed to resolve exactly three experiments'
printf 'selected_experiment_indices=%s %s %s\n' \
    "${TASK_EXPERIMENT_INDICES[0]}" \
    "${TASK_EXPERIMENT_INDICES[1]}" \
    "${TASK_EXPERIMENT_INDICES[2]}"
echo 'experiment_order=ranklift_tied_c124_m460 then product_code_quota_h6144 then btmos_k3_c256_lb'
echo 'precision=bf16 seed=42 block_size=2048'
echo 'batch=16_per_device_x_4_accumulation_x_8_gpus=512_sequences_per_update'
echo 'schedule=one_epoch_with_runner_forced_stop_at_checkpoint_10000'
echo 'ddp_find_unused_parameters=false for all three custom tied architectures'

mkdir -p "$TASK_LOG_DIR"
echo '=== start three sequential 10k training experiments ==='
"$TASK_PYTHON" -u run_experiments.py \
    --experiments "${TASK_EXPERIMENT_INDICES[@]}" \
    --stop-at-step 10000 \
    --log-dir "$TASK_LOG_DIR"

echo '=== sleep 30 seconds after sequential training ==='
sleep 30
require_free_gpus '30 SECONDS AFTER ALL THREE TRAINING RUNS'

echo '=== validate all checkpoint-10000 artifacts and logs ==='
"$TASK_PYTHON" - "$TASK_LOG_DIR" "${TASK_OUTPUT_DIRS[@]}" <<'PY'
import json
import math
import pathlib
import sys

log_dir = pathlib.Path(sys.argv[1])
output_dirs = [pathlib.Path(value) for value in sys.argv[2:]]
required = (
    "config.json", "model.safetensors", "trainer_state.json",
    "optimizer.pt", "scheduler.pt", "embedding.pt",
    "rng_state_0.pth", "rng_state_1.pth", "rng_state_2.pth",
    "rng_state_3.pth", "rng_state_4.pth", "rng_state_5.pth",
    "rng_state_6.pth", "rng_state_7.pth",
)
fatal = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OutOfMemoryError",
    "ChildFailedError",
    "ProcessExitedException",
    "Segmentation fault",
    "Bus error",
)
for output_dir in output_dirs:
    checkpoint = output_dir / "checkpoint-10000"
    missing = [
        name for name in required
        if not (checkpoint / name).is_file()
        or (checkpoint / name).stat().st_size == 0
    ]
    assert not missing, (str(checkpoint), missing)
    with (checkpoint / "trainer_state.json").open() as handle:
        state = json.load(handle)
    assert int(state["global_step"]) == 10_000, (
        output_dir.name, state["global_step"]
    )
    losses = [
        float(row["loss"])
        for row in state.get("log_history", [])
        if "loss" in row
    ]
    assert losses and all(math.isfinite(value) for value in losses), (
        output_dir.name, losses[-10:]
    )
    log_path = log_dir / f"{output_dir.name}.log"
    text = log_path.read_text(errors="replace")
    found = [marker for marker in fatal if marker in text]
    assert not found, (output_dir.name, found)
    print(
        f"CHECKPOINT_10000_VALID name={output_dir.name} "
        f"logged_losses={len(losses)} first={losses[0]:.6f} "
        f"last={losses[-1]:.6f}"
    )

summary = (log_dir / "experiments.log").read_text(errors="replace")
for output_dir in output_dirs:
    expected = f"{output_dir.name}: STOPPED at step 10000"
    assert expected in summary, expected
print("ALL_THREE_EXPERIMENT_LOG_STATUSES_VALID")
PY

echo '=== atomically write reusable successful-training marker ==='
mkdir -p "$TASK_STATUS_DIR"
{
    printf 'STATUS=SUCCESS\n'
    printf 'COMPLETED_UTC=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf 'STOP_STEP=10000\n'
    printf 'EXPERIMENT_1=ranklift_tied_c124_m460\n'
    printf 'EXPERIMENT_2=product_code_quota_h6144\n'
    printf 'EXPERIMENT_3=btmos_k3_c256_lb\n'
    printf 'CHECKPOINT_1=%s\n' "${TASK_OUTPUT_DIRS[0]}/checkpoint-10000"
    printf 'CHECKPOINT_2=%s\n' "${TASK_OUTPUT_DIRS[1]}/checkpoint-10000"
    printf 'CHECKPOINT_3=%s\n' "${TASK_OUTPUT_DIRS[2]}/checkpoint-10000"
    printf 'LOG_DIR=%s\n' "$TASK_LOG_DIR"
} >"$TASK_COMPLETION_TMP"
mv "$TASK_COMPLETION_TMP" "$TASK_COMPLETION_FILE"
test -s "$TASK_COMPLETION_FILE"
cat "$TASK_COMPLETION_FILE"
echo "TRAINING_COMPLETION_MARKER=$TASK_COMPLETION_FILE"

echo '=== start and verify runner-provided burn after successful training ==='
start_burn_and_verify "$TASK_BURN_LOG"
TASK_BURN_RESTORED=1
echo 'TH2 RANKLIFT HASHEDV2 BTMOS TRAINING COMPLETE AND BURNS ACTIVE'
