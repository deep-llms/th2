#1 +300+a
#th2-train-tiered-c512-and-lb-groupreduce-10k-20260904-a01
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
    nvidia-smi \
        --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv,noheader
    [[ "${#TASK_STAGE_GPU_PIDS[@]}" -eq 0 ]] \
        || die "GPU compute processes remain $stage: ${TASK_STAGE_GPU_PIDS[*]}"
    echo "ALL 8 B200 GPUS FREE $stage"
}

require_burn_ancestry() {
    local worker_pid="$1"
    local current_pid="$worker_pid"
    local parent_pid cmdline depth
    local found=0
    for depth in 1 2 3 4 5 6; do
        [[ -r "/proc/$current_pid/status" ]] \
            || die "cannot inspect burn worker ancestry for PID $worker_pid"
        parent_pid="$(awk '/^PPid:/ {print $2}' "/proc/$current_pid/status")"
        [[ "$parent_pid" =~ ^[0-9]+$ ]] \
            || die "invalid parent PID while checking burn worker $worker_pid"
        # PID 1 embeds the burn path in the runner sleeper command line. Never
        # accept PID 1 as ownership evidence and never signal it.
        [[ "$parent_pid" -ne 1 ]] || break
        [[ -r "/proc/$parent_pid/cmdline" ]] \
            || die "cannot read parent $parent_pid for burn worker $worker_pid"
        cmdline="$(tr '\0' ' ' < "/proc/$parent_pid/cmdline")"
        echo "burn_ancestry worker=$worker_pid depth=$depth parent=$parent_pid cmdline=$cmdline"
        if [[ "$cmdline" == *'/tmp/llm_pretrain_burn.py'* ]]; then
            found=1
            break
        fi
        current_pid="$parent_pid"
    done
    [[ "$found" -eq 1 ]] \
        || die "GPU PID $worker_pid has no non-PID-1 runner-burn ancestor"
}

verify_one_process_per_gpu() {
    local expected_kind="$1"
    local -a per_gpu_pids unique_pids
    local gpu_index output
    per_gpu_pids=()
    for gpu_index in 0 1 2 3 4 5 6 7; do
        output="$(
            nvidia-smi -i "$gpu_index" \
                --query-compute-apps=pid --format=csv,noheader,nounits
        )" || die "failed to inspect GPU $gpu_index"
        mapfile -t TASK_ONE_GPU_PIDS < <(
            printf '%s\n' "$output" |
                sed 's/^[[:space:]]*//;s/[[:space:]]*$//;/^$/d' | sort -nu
        )
        [[ "${#TASK_ONE_GPU_PIDS[@]}" -eq 1 ]] \
            || die "GPU $gpu_index does not have exactly one $expected_kind process"
        [[ "${TASK_ONE_GPU_PIDS[0]}" -ne 1 ]] || die 'refusing PID 1'
        kill -0 "${TASK_ONE_GPU_PIDS[0]}" \
            || die "$expected_kind process disappeared on GPU $gpu_index"
        per_gpu_pids+=("${TASK_ONE_GPU_PIDS[0]}")
        echo "gpu=$gpu_index kind=$expected_kind pid=${TASK_ONE_GPU_PIDS[0]}"
    done
    mapfile -t unique_pids < <(printf '%s\n' "${per_gpu_pids[@]}" | sort -nu)
    [[ "${#unique_pids[@]}" -eq 8 ]] \
        || die "the eight GPUs do not have eight distinct $expected_kind processes"
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
    TASK_NEW_BURN_PIDS=()
    read_gpu_pids TASK_NEW_BURN_PIDS || return 1
    [[ "${#TASK_NEW_BURN_PIDS[@]}" -eq 8 ]] || {
        echo "ERROR: expected 8 burn workers, found ${#TASK_NEW_BURN_PIDS[@]}" >&2
        nvidia-smi
        return 1
    }
    verify_one_process_per_gpu burn
    for TASK_PID in "${TASK_NEW_BURN_PIDS[@]}"; do
        require_burn_ancestry "$TASK_PID"
    done
    nvidia-smi \
        --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv,noheader
    echo 'ALL 8 RUNNER GPU BURNS ACTIVE'
}

restore_burn_on_exit() {
    local rc=$?
    trap - EXIT
    set +e
    if [[ "${TASK_NEEDS_BURN_RESTORE:-0}" -eq 1 ]]; then
        TASK_EXIT_GPU_PIDS=()
        if ! read_gpu_pids TASK_EXIT_GPU_PIDS; then
            echo 'ERROR: cannot query GPU processes during exit; burn not started' >&2
        elif [[ "${#TASK_EXIT_GPU_PIDS[@]}" -eq 0 ]]; then
            echo "=== restoring runner burn during exit rc=$rc ==="
            start_burn_and_verify "$TASK_BURN_LOG" \
                || echo 'ERROR: could not restore runner burn during exit' >&2
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
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/tiered_c512_lb_groupreduce_10k_20260904_a01"
TASK_STATUS_DIR="$TASK_OUTPUT_BASE/status"
TASK_COMPLETION_FILE="$TASK_STATUS_DIR/tiered_c512_lb_groupreduce_10k_20260904_a01.complete"
TASK_COMPLETION_TMP="$TASK_COMPLETION_FILE.tmp"
TASK_BURN_LOG="$TASK_OUTPUT_BASE/logs/gpu_burn_after_tiered_c512_lb_groupreduce_20260904_a01.log"
TASK_ACCELERATE_SOURCE="$TASK_PROJECT_DIR/resources/accelerate_config.yaml"
TASK_ACCELERATE_TARGET=/mnt/local/.cache/huggingface/accelerate/default_config.yaml
TASK_OUTPUT_DIRS=(
    "$TASK_OUTPUT_BASE/tiered_ranklift_lb_t4_c512"
    "$TASK_OUTPUT_BASE/groupreduce_matched_lb_t4"
)
TASK_LAUNCHERS=(
    scripts/train_tiered_ranklift_lb_tied.sh
    scripts/train_groupreduce_matched_lb_tied.sh
)
TASK_NEEDS_BURN_RESTORE=0
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
echo '39b15eab8cf213d563dcf5137bb982e836bb8e3beba8e7def8dcddf21fe43594  resources/token_importance_langbalanced.npz' \
    | sha256sum -c -
echo 'a086dcdc7bfe6d8376755e63df796a0e60157110b04ba691dc5c3ad2ff275029  resources/token_importance_langbalanced.json' \
    | sha256sum -c -
echo '923db7f20a2df3d051180f67f9bea1f30c84c804651e313fa9961a9fd17a57e5  resources/accelerate_config.yaml' \
    | sha256sum -c -
for TASK_LAUNCHER in "${TASK_LAUNCHERS[@]}"; do
    test -s "$TASK_LAUNCHER"
    bash -n "$TASK_LAUNCHER"
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
        die "launcher must retain one epoch; stop only through the runner"
    fi
done
grep -Fq -- '--tiered_ranklift_code_dims 1024,512,192,64' \
    scripts/train_tiered_ranklift_lb_tied.sh
grep -Fq -- '--tiered_ranklift_lift_dims 0,0,320,192' \
    scripts/train_tiered_ranklift_lb_tied.sh
grep -Fq -- '--groupreduce_ranks 1024,512,192,64' \
    scripts/train_groupreduce_matched_lb_tied.sh
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

"$TASK_PYTHON" - <<'PY'
from compositional.compressed_baselines import GroupReduceEmbed
from compositional.compression_init import (
    frequency_group_ids_from_populations,
    load_frequency_counts,
)
from compositional.nonlinear_factorizations import TieredRankLiftEmbed

vocab = 151_936
dim = 1_024
populations = (2_048, 6_144, 24_576, 119_168)
importance = load_frequency_counts(
    "resources/token_importance_langbalanced.npz",
    vocab,
    key="counts",
    pseudocount=0.0,
)
groups = frequency_group_ids_from_populations(importance, populations)
control = GroupReduceEmbed(
    vocab, dim, group_ranks=(1_024, 512, 192, 64), group_ids=groups
)
tiered = TieredRankLiftEmbed(
    vocab,
    dim,
    code_dims=(1_024, 512, 192, 64),
    lift_dims=(0, 0, 320, 192),
    group_ids=groups,
)
control_count = sum(parameter.numel() for parameter in control.parameters())
tiered_count = sum(parameter.numel() for parameter in tiered.parameters())
assert control_count == 19_423_232, control_count
assert tiered_count == 20_096_000, tiered_count
assert tiered_count - control_count == 672_768
assert control.group_sizes == tiered.group_sizes == populations
print(
    "INTERFACE_CONFIG_OK",
    f"control_params={control_count}",
    f"tiered_params={tiered_count}",
    f"delta={tiered_count - control_count}",
)
PY
require_b200_node

echo '=== prove no real project workload precedes burn reclamation ==='
if pgrep -af '[r]un_experiments.py|[t]rain_compositional.py|[t]rain.py|[e]val_parallel.py|[f]inetune/run_all.py|[f]inetune/train.py|[p]repare_data.py'; then
    die 'a real training, evaluation, finetuning, or preparation process exists'
fi
TASK_BURN_PIDS=()
read_gpu_pids TASK_BURN_PIDS || die 'failed to query current GPU compute PIDs'
[[ "${#TASK_BURN_PIDS[@]}" -eq 8 ]] \
    || die "expected exactly 8 current burn workers, found ${#TASK_BURN_PIDS[@]}"
verify_one_process_per_gpu burn
for TASK_PID in "${TASK_BURN_PIDS[@]}"; do
    [[ "$TASK_PID" =~ ^[0-9]+$ ]] || die "invalid GPU PID: $TASK_PID"
    [[ "$TASK_PID" -ne 1 ]] || die 'refusing to signal PID 1'
    require_burn_ancestry "$TASK_PID"
done
nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader

echo '=== stop only the eight verified GPU burn workers ==='
kill -9 "${TASK_BURN_PIDS[@]}" 2>/dev/null || true
TASK_NEEDS_BURN_RESTORE=1
echo '=== wait 30 seconds after burn stop ==='
sleep 30
require_free_gpus '30 SECONDS AFTER BURN STOP'

echo '=== copy and verify exact 8-GPU BF16 Accelerate config ==='
mkdir -p "$(dirname "$TASK_ACCELERATE_TARGET")"
cp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
cmp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
grep -Fxq 'distributed_type: MULTI_GPU' "$TASK_ACCELERATE_TARGET"
grep -Fxq 'mixed_precision: bf16' "$TASK_ACCELERATE_TARGET"
grep -Fxq 'num_processes: 8' "$TASK_ACCELERATE_TARGET"
echo '=== wait 30 seconds after Accelerate config copy ==='
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
unset BTMOS_OUTPUT_DIR BTMOS_IMPORTANCE_PATH || true

echo '=== resolve exact experiments by name ==='
mapfile -t TASK_EXPERIMENT_INDICES < <(
    "$TASK_PYTHON" - <<'PY'
from run_experiments import EXPERIMENT_COMMANDS

expected = (
    (
        "tiered_ranklift_lb_t4_c512",
        "bash scripts/train_tiered_ranklift_lb_tied.sh",
    ),
    (
        "groupreduce_matched_lb_t4",
        "bash scripts/train_groupreduce_matched_lb_tied.sh",
    ),
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
[[ "${#TASK_EXPERIMENT_INDICES[@]}" -eq 2 ]] \
    || die 'failed to resolve exactly two experiments'
printf 'selected_experiment_indices=%s %s\n' \
    "${TASK_EXPERIMENT_INDICES[0]}" "${TASK_EXPERIMENT_INDICES[1]}"
echo 'experiment_order=tiered_ranklift_lb_t4_c512 then groupreduce_matched_lb_t4'
echo 'precision=bf16 seed=42 block_size=2048'
echo 'batch=16_per_device_x_4_accumulation_x_8_gpus=512_sequences_per_update'
echo 'schedule=one_epoch_with_runner_forced_stop_at_checkpoint_10000'
echo 'ddp_find_unused_parameters=false and exact tied output for both arms'

mkdir -p "$TASK_LOG_DIR"
echo '=== start two sequential 10k training experiments ==='
"$TASK_PYTHON" -u run_experiments.py \
    --experiments "${TASK_EXPERIMENT_INDICES[@]}" \
    --stop-at-step 10000 \
    --log-dir "$TASK_LOG_DIR"

echo '=== wait 30 seconds after sequential training ==='
sleep 30
require_free_gpus '30 SECONDS AFTER BOTH TRAINING RUNS'

echo '=== validate both checkpoint-10000 saves and training logs ==='
"$TASK_PYTHON" - "$TASK_LOG_DIR" "${TASK_OUTPUT_DIRS[@]}" <<'PY'
import json
import math
import pathlib
import sys

log_dir = pathlib.Path(sys.argv[1])
output_dirs = [pathlib.Path(value) for value in sys.argv[2:]]
assert len(output_dirs) == 2
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
    config = json.loads((output_dir / "train_config.json").read_text())
    comp = config["compositional"]
    training = config["training"]
    assert comp["tie_output"] is True, output_dir
    assert training["ddp_find_unused_parameters"] is False, output_dir
    if output_dir.name == "tiered_ranklift_lb_t4_c512":
        assert comp["arm"] == "tiered_ranklift", comp
        assert comp["tiered_ranklift_code_dims"] == "1024,512,192,64"
        assert comp["tiered_ranklift_lift_dims"] == "0,0,320,192"
    else:
        assert comp["arm"] == "groupreduce", comp
        assert comp["groupreduce_ranks"] == "1024,512,192,64"
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
print("BOTH_EXPERIMENT_LOG_STATUSES_VALID")
PY

echo '=== atomically write reusable successful-training marker ==='
mkdir -p "$TASK_STATUS_DIR"
{
    printf 'STATUS=SUCCESS\n'
    printf 'COMPLETED_UTC=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf 'STOP_STEP=10000\n'
    printf 'EXPERIMENT_1=tiered_ranklift_lb_t4_c512\n'
    printf 'EXPERIMENT_2=groupreduce_matched_lb_t4\n'
    printf 'CHECKPOINT_1=%s\n' "${TASK_OUTPUT_DIRS[0]}/checkpoint-10000"
    printf 'CHECKPOINT_2=%s\n' "${TASK_OUTPUT_DIRS[1]}/checkpoint-10000"
    printf 'LOG_DIR=%s\n' "$TASK_LOG_DIR"
} >"$TASK_COMPLETION_TMP"
mv "$TASK_COMPLETION_TMP" "$TASK_COMPLETION_FILE"
test -s "$TASK_COMPLETION_FILE"
cat "$TASK_COMPLETION_FILE"
echo "TRAINING_COMPLETION_MARKER=$TASK_COMPLETION_FILE"

echo '=== start and verify runner-provided burn after successful training ==='
start_burn_and_verify "$TASK_BURN_LOG"
TASK_NEEDS_BURN_RESTORE=0
echo 'TH2 TIERED-C512 AND LANGUAGE-BALANCED GROUPREDUCE TRAINING COMPLETE; BURNS ACTIVE'
