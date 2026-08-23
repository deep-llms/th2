#1 +120+a
#th2-verify-completed-shared-local-g16-and-launch-gpu-burn-20260823
#!/usr/bin/env bash
set -euo pipefail

echo '=== supervise the current SharedLocal G16 run, then launch GPU burns ==='
date -u
hostname

TASK_OUTPUT=/mnt/local/_outputs/@PROJECT@/shared_local_tied_g16
TASK_CHECKPOINT="$TASK_OUTPUT/checkpoint-10000"
TASK_TRAIN_CONFIG="$TASK_OUTPUT/train_config.json"
TASK_LOG_DIR=/mnt/local/_outputs/@PROJECT@/logs/shared_local_tied_g16
TASK_EXPERIMENT_LOG="$TASK_LOG_DIR/experiments.log"
TASK_TRAIN_LOG="$TASK_LOG_DIR/shared_local_tied_g16.log"
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_BURN_PYTHON=/usr/bin/python3
TASK_BURN_SCRIPT="$PWD/scripts/gpu_burn.py"

die() {
    echo "ERROR: $*"
    exit 1
}

[[ -x "$TASK_PYTHON" ]] || die "missing training Python: $TASK_PYTHON"
[[ -x "$TASK_BURN_PYTHON" ]] || die "missing burn Python: $TASK_BURN_PYTHON"
[[ -f "$TASK_BURN_SCRIPT" && ! -L "$TASK_BURN_SCRIPT" && -s "$TASK_BURN_SCRIPT" ]] \
    || die "missing, empty, or non-regular burn script: $TASK_BURN_SCRIPT"
[[ -d "$TASK_OUTPUT" && ! -L "$TASK_OUTPUT" ]] \
    || die "missing or non-directory output: $TASK_OUTPUT"
[[ -d "$TASK_LOG_DIR" && ! -L "$TASK_LOG_DIR" ]] \
    || die "missing or non-directory log directory: $TASK_LOG_DIR"
for TASK_FILE in "$TASK_EXPERIMENT_LOG" "$TASK_TRAIN_LOG" "$TASK_TRAIN_CONFIG"; do
    [[ -f "$TASK_FILE" && ! -L "$TASK_FILE" && -s "$TASK_FILE" ]] \
        || die "missing, empty, or non-regular file: $TASK_FILE"
done

trim() {
    local value=$1
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

proc_snapshot() {
    local pid=$1
    local stat rest
    [[ -r "/proc/$pid/stat" ]] || return 1
    stat=$(<"/proc/$pid/stat") || return 1
    [[ "$stat" == *') '* ]] || return 1
    rest=${stat#*) }
    TASK_PROC_STATE=${rest%% *}
    TASK_PROC_START_TICKS=$(awk '{print $20}' <<<"$rest")
    [[ "$TASK_PROC_START_TICKS" =~ ^[0-9]+$ ]] || return 1
}

same_proc() {
    local pid=$1
    local expected_start=$2
    proc_snapshot "$pid" || return 1
    [[ "$TASK_PROC_START_TICKS" == "$expected_start" ]] || return 1
    [[ "$TASK_PROC_STATE" != Z && "$TASK_PROC_STATE" != X ]]
}

read_proc_cmdline() {
    local pid=$1
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    TASK_PROC_CMDLINE=$(tr '\0' ' ' <"/proc/$pid/cmdline") || return 1
    [[ -n "$TASK_PROC_CMDLINE" ]]
}

find_exact_runner() {
    local proc_dir pid exe
    TASK_RUNNER_MATCHES=()
    for proc_dir in /proc/[0-9]*; do
        pid=${proc_dir##*/}
        proc_snapshot "$pid" || continue
        [[ "$TASK_PROC_STATE" != Z && "$TASK_PROC_STATE" != X ]] || continue
        read_proc_cmdline "$pid" || continue
        exe=$(readlink -f "$proc_dir/exe" 2>/dev/null || true)
        [[ "$exe" == *python* ]] || continue
        case "$TASK_PROC_CMDLINE" in
            *"run_experiments.py --experiments 11 --stop-at-step 10000 --log-dir $TASK_LOG_DIR"*)
                TASK_RUNNER_MATCHES+=("$pid")
                ;;
        esac
    done
}

find_exact_training_processes() {
    local proc_dir pid exe
    TASK_TRAIN_MATCHES=()
    for proc_dir in /proc/[0-9]*; do
        pid=${proc_dir##*/}
        proc_snapshot "$pid" || continue
        [[ "$TASK_PROC_STATE" != Z && "$TASK_PROC_STATE" != X ]] || continue
        read_proc_cmdline "$pid" || continue
        exe=$(readlink -f "$proc_dir/exe" 2>/dev/null || true)
        [[ "$exe" == *python* ]] || continue
        case "$TASK_PROC_CMDLINE" in
            *train_compositional.py*"--output_dir $TASK_OUTPUT"*"--run_name shared-local-tied-g16-qwen3-0.6b"*"--arm shared_local"*"--shared_rank 64"*"--local_embed_rank 64"*"--num_groups 16"*"--tie_output"*)
                TASK_TRAIN_MATCHES+=("$pid")
                ;;
        esac
    done
}

runner_owns_experiment_log() {
    local pid=$1
    local expected_identity fd actual_identity
    expected_identity=$(stat -Lc '%d:%i' "$TASK_EXPERIMENT_LOG") || return 1
    for fd in "/proc/$pid/fd/"*; do
        actual_identity=$(stat -Lc '%d:%i' "$fd" 2>/dev/null || true)
        [[ "$actual_identity" == "$expected_identity" ]] && return 0
    done
    return 1
}

query_compute_pairs() {
    local raw uuid pid
    raw=$(nvidia-smi \
        --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits) \
        || die 'failed to query GPU compute processes'
    TASK_COMPUTE_PAIRS=()
    while IFS=, read -r uuid pid; do
        uuid=$(trim "${uuid:-}")
        pid=$(trim "${pid:-}")
        [[ -z "$uuid" && -z "$pid" ]] && continue
        [[ -n "$uuid" ]] || die "GPU process query returned an empty UUID: $raw"
        [[ "$pid" =~ ^[0-9]+$ ]] \
            || die "GPU process query returned an invalid PID: $pid"
        TASK_COMPUTE_PAIRS+=("$uuid,$pid")
    done <<<"$raw"
}

completion_failed() {
    grep -Fq 'shared_local_tied_g16 — FAILED' "$TASK_EXPERIMENT_LOG" \
        || grep -Fq 'experiment(s) did not complete successfully' "$TASK_EXPERIMENT_LOG"
}

completion_succeeded() {
    grep -Fq 'DONE: shared_local_tied_g16 — STOPPED at step 10000' \
        "$TASK_EXPERIMENT_LOG" \
        && grep -Fq 'All 1 experiments done' "$TASK_EXPERIMENT_LOG"
}

echo '=== verify the expected eight-B200 inventory ==='
TASK_GPU_INVENTORY_RAW=$(nvidia-smi \
    --query-gpu=index,name,uuid --format=csv,noheader) \
    || die 'failed to query GPU inventory'
TASK_GPU_INDICES=()
TASK_EXPECTED_GPU_UUIDS=()
declare -A TASK_GPU_INDEX_BY_UUID=()
declare -A TASK_GPU_NAME_BY_UUID=()
while IFS=, read -r index name uuid; do
    index=$(trim "${index:-}")
    name=$(trim "${name:-}")
    uuid=$(trim "${uuid:-}")
    [[ "$index" =~ ^[0-9]+$ ]] || die "invalid GPU index: $index"
    [[ "$name" == *B200* ]] || die "GPU $index is not a B200: $name"
    [[ -n "$uuid" ]] || die "GPU $index has no UUID"
    [[ -z "${TASK_GPU_INDEX_BY_UUID[$uuid]+x}" ]] \
        || die "duplicate GPU UUID: $uuid"
    TASK_GPU_INDICES+=("$index")
    TASK_EXPECTED_GPU_UUIDS+=("$uuid")
    TASK_GPU_INDEX_BY_UUID[$uuid]=$index
    TASK_GPU_NAME_BY_UUID[$uuid]=$name
    echo "gpu=$index name=$name uuid=$uuid"
done <<<"$TASK_GPU_INVENTORY_RAW"
[[ "${#TASK_EXPECTED_GPU_UUIDS[@]}" -eq 8 ]] \
    || die "expected 8 B200 GPUs, found ${#TASK_EXPECTED_GPU_UUIDS[@]}"

echo '=== require the SharedLocal run to have exited cleanly ==='
find_exact_runner
[[ "${#TASK_RUNNER_MATCHES[@]}" -eq 0 ]] \
    || die "the exact G16 runner is still active: ${TASK_RUNNER_MATCHES[*]}"
find_exact_training_processes
[[ "${#TASK_TRAIN_MATCHES[@]}" -eq 0 ]] \
    || die "matching G16 training processes are still active: ${TASK_TRAIN_MATCHES[*]}"

[[ "$(grep -Fc '[1/1] shared_local_tied_g16' "$TASK_EXPERIMENT_LOG" || true)" -eq 1 ]] \
    || die 'experiment log does not contain exactly one G16 selection marker'
[[ "$(grep -Fc 'START: shared_local_tied_g16' "$TASK_EXPERIMENT_LOG" || true)" -eq 1 ]] \
    || die 'experiment log does not contain exactly one G16 start marker'
[[ "$(grep -Fc 'Will stop at step 10000' "$TASK_EXPERIMENT_LOG" || true)" -eq 1 ]] \
    || die 'experiment log does not contain exactly one stop-step marker'
if completion_failed; then
    tail -100 "$TASK_EXPERIMENT_LOG"
    die 'the experiment runner reported failure'
fi
if ! completion_succeeded; then
    tail -100 "$TASK_EXPERIMENT_LOG"
    die 'the experiment runner has no successful checkpoint-10000 completion marker'
fi
echo 'runner_completion_marker_seen=true'

echo '=== verify checkpoint-10000 is the latest complete checkpoint ==='
TASK_CHECKPOINTS_RAW=$(find "$TASK_OUTPUT" -mindepth 1 -maxdepth 1 \
    -type d -name 'checkpoint-*' -printf '%f\n') \
    || die 'failed to enumerate final checkpoints'
TASK_LATEST_CHECKPOINT=$(printf '%s\n' "$TASK_CHECKPOINTS_RAW" \
    | awk 'NF' | sort -V | tail -1)
[[ "$TASK_LATEST_CHECKPOINT" == checkpoint-10000 ]] \
    || die "latest checkpoint is ${TASK_LATEST_CHECKPOINT:-none}, expected checkpoint-10000"
[[ -d "$TASK_CHECKPOINT" && ! -L "$TASK_CHECKPOINT" ]] \
    || die "checkpoint is missing, symlinked, or not a directory: $TASK_CHECKPOINT"
for TASK_REQUIRED in config.json model.safetensors trainer_state.json \
        optimizer.pt scheduler.pt embedding.pt rng_state_0.pth rng_state_1.pth \
        rng_state_2.pth rng_state_3.pth rng_state_4.pth rng_state_5.pth \
        rng_state_6.pth rng_state_7.pth; do
    TASK_REQUIRED_PATH="$TASK_CHECKPOINT/$TASK_REQUIRED"
    [[ -f "$TASK_REQUIRED_PATH" && ! -L "$TASK_REQUIRED_PATH" \
        && -s "$TASK_REQUIRED_PATH" ]] \
        || die "missing, empty, symlinked, or non-regular checkpoint artifact: $TASK_REQUIRED_PATH"
done

"$TASK_PYTHON" - "$TASK_CHECKPOINT" "$TASK_TRAIN_CONFIG" \
    "$TASK_TRAIN_LOG" <<'PY'
import gc
import json
import math
import os
import re
import sys
import zipfile
from collections import Counter

import torch

checkpoint_dir, train_config_path, log_path = sys.argv[1:]


def require(condition, message):
    if not condition:
        raise SystemExit(f"ERROR: {message}")


def walk_tensors(value, path="root"):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from walk_tensors(child, f"{path}[{key!r}]")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from walk_tensors(child, f"{path}[{index}]")


def check_torch_archive(path):
    require(zipfile.is_zipfile(path), f"not a valid PyTorch ZIP archive: {path}")
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
    require(bad_member is None, f"CRC failure in {path}: {bad_member}")


state_path = os.path.join(checkpoint_dir, "trainer_state.json")
with open(state_path, encoding="utf-8") as handle:
    trainer_state = json.load(handle)
global_step = trainer_state.get("global_step")
require(
    isinstance(global_step, int) and not isinstance(global_step, bool)
    and global_step == 10_000,
    f"trainer global_step={global_step!r}, expected 10000",
)
metrics = [
    row for row in trainer_state.get("log_history", [])
    if isinstance(row, dict) and "loss" in row
]
expected_steps = list(range(10, 10_001, 10))
actual_steps = [row.get("step") for row in metrics]
require(
    actual_steps == expected_steps,
    f"loss metric steps are incomplete, duplicated, or unordered: "
    f"count={len(actual_steps)}, final={actual_steps[-1:]}",
)
for row in metrics:
    for key in ("loss", "grad_norm", "learning_rate"):
        require(key in row, f"metric at step {row['step']} has no {key}")
        require(
            math.isfinite(float(row[key])),
            f"non-finite {key}={row[key]!r} at step {row['step']}",
        )

with open(train_config_path, encoding="utf-8") as handle:
    train_config = json.load(handle)
comp = train_config.get("compositional")
require(isinstance(comp, dict), "train_config.json has no compositional mapping")
expected_comp = {
    "arm": "shared_local",
    "shared_rank": 64,
    "local_embed_rank": 64,
    "num_groups": 16,
    "tie_output": True,
    "independent_lowrank_output": False,
}
for key, expected in expected_comp.items():
    require(
        comp.get(key) == expected,
        f"train config {key}={comp.get(key)!r}, expected {expected!r}",
    )

for filename in (
    "embedding.pt",
    "optimizer.pt",
    "scheduler.pt",
    *(f"rng_state_{rank}.pth" for rank in range(8)),
):
    check_torch_archive(os.path.join(checkpoint_dir, filename))

from compositional.loading import load_compositional_model
from compositional.tied_head import TiedSharedLocalHead

model, loaded_comp = load_compositional_model(
    checkpoint_dir, device="cpu", dtype=torch.bfloat16
)
for key, expected in expected_comp.items():
    require(
        loaded_comp.get(key) == expected,
        f"loaded config {key}={loaded_comp.get(key)!r}, expected {expected!r}",
    )
require(
    isinstance(model.lm_head, TiedSharedLocalHead),
    f"loaded output head has type {type(model.lm_head).__name__}",
)
require(
    model.lm_head.embed is model.model.embed_tokens.embed,
    "loaded output head is not exactly tied to the input embedding",
)
require(model.config.model_type == "qwen3", f"model_type={model.config.model_type!r}")
require(model.config.vocab_size == 151_936, f"vocab_size={model.config.vocab_size!r}")
require(model.config.hidden_size == 1_024, f"hidden_size={model.config.hidden_size!r}")
require(
    model.config.tie_word_embeddings is False,
    f"tie_word_embeddings={model.config.tie_word_embeddings!r}, expected False",
)

expected_state_shapes = {
    key: tuple(value.shape) for key, value in model.state_dict().items()
}
expected_parameter_shapes = Counter(
    tuple(parameter.shape)
    for parameter in model.parameters()
    if parameter.requires_grad
)
from safetensors import safe_open

model_path = os.path.join(checkpoint_dir, "model.safetensors")
with safe_open(model_path, framework="pt", device="cpu") as handle:
    actual_state_shapes = {
        key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()
    }
    require(
        actual_state_shapes == expected_state_shapes,
        "model.safetensors key/shape topology differs from the exact rebuilt model: "
        f"missing={sorted(set(expected_state_shapes) - set(actual_state_shapes))}, "
        f"extra={sorted(set(actual_state_shapes) - set(expected_state_shapes))}, "
        f"shape_mismatches={sorted(key for key in expected_state_shapes.keys() & actual_state_shapes.keys() if expected_state_shapes[key] != actual_state_shapes[key])}",
    )
    require(
        not any(key.startswith("lm_head.") for key in handle.keys()),
        "an exactly tied SharedLocal checkpoint must not save lm_head parameters",
    )
    for key in handle.keys():
        tensor = handle.get_tensor(key)
        if tensor.is_floating_point() or tensor.is_complex():
            require(
                torch.isfinite(tensor).all().item(),
                f"non-finite saved model tensor: {key}",
            )
        del tensor

model_tensor_count = 0
model_parameter_count = 0
for name, parameter in model.named_parameters():
    model_tensor_count += 1
    model_parameter_count += parameter.numel()
    require(
        torch.isfinite(parameter).all().item(),
        f"non-finite model parameter: {name}",
    )
require(model_tensor_count > 0, "loaded model has no parameters")
del model
gc.collect()

optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
optimizer_state = torch.load(
    optimizer_path, map_location="cpu", weights_only=True, mmap=True
)
require(isinstance(optimizer_state, dict), "optimizer.pt is not a mapping")
require(
    set(optimizer_state) == {"state", "param_groups"},
    f"optimizer.pt has unexpected top-level keys: {sorted(optimizer_state)}",
)
require(bool(optimizer_state.get("state")), "optimizer.pt has no parameter state")
require(
    isinstance(optimizer_state.get("param_groups"), list)
    and bool(optimizer_state["param_groups"]),
    "optimizer.pt has no parameter groups",
)
optimizer_groups = optimizer_state["param_groups"]
flat_parameter_ids = [
    parameter_id
    for group in optimizer_groups
    for parameter_id in group.get("params", [])
]
require(
    len(flat_parameter_ids) == len(set(flat_parameter_ids)),
    "optimizer parameter IDs are duplicated across parameter groups",
)
require(
    set(flat_parameter_ids) == set(optimizer_state["state"]),
    "optimizer parameter groups and state contain different parameter IDs",
)
for group_index, group in enumerate(optimizer_groups):
    for key in ("lr", "eps", "weight_decay"):
        require(
            key in group and math.isfinite(float(group[key])),
            f"optimizer group {group_index} has invalid {key}={group.get(key)!r}",
        )
    betas = group.get("betas")
    require(
        isinstance(betas, (list, tuple)) and len(betas) == 2
        and all(math.isfinite(float(beta)) for beta in betas),
        f"optimizer group {group_index} has invalid betas={betas!r}",
    )

optimizer_tensor_count = 0
optimizer_parameter_shapes = Counter()
for parameter_id, slot in optimizer_state["state"].items():
    require(isinstance(slot, dict), f"optimizer slot {parameter_id!r} is not a mapping")
    for key in ("step", "exp_avg", "exp_avg_sq"):
        require(key in slot, f"optimizer slot {parameter_id!r} has no {key}")
        require(
            isinstance(slot[key], torch.Tensor),
            f"optimizer slot {parameter_id!r} {key} is not a tensor",
        )
    step = slot["step"]
    exp_avg = slot["exp_avg"]
    exp_avg_sq = slot["exp_avg_sq"]
    require(step.numel() == 1 and int(step.item()) == 10_000,
            f"optimizer slot {parameter_id!r} step={step}")
    require(exp_avg.shape == exp_avg_sq.shape,
            f"optimizer slot {parameter_id!r} moment shapes differ")
    optimizer_parameter_shapes[tuple(exp_avg.shape)] += 1
    for key, tensor in slot.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        optimizer_tensor_count += 1
        if tensor.is_floating_point() or tensor.is_complex():
            require(
                torch.isfinite(tensor).all().item(),
                f"non-finite optimizer tensor: state[{parameter_id!r}][{key!r}]",
            )
    require(
        torch.all(exp_avg_sq >= 0).item(),
        f"optimizer slot {parameter_id!r} has a negative second moment",
    )
require(optimizer_tensor_count > 0, "optimizer.pt contains no tensors")
require(
    optimizer_parameter_shapes == expected_parameter_shapes,
    "optimizer moment shapes do not exactly cover the trainable model parameters",
)
optimizer_lrs = [float(group["lr"]) for group in optimizer_groups]
del optimizer_state
gc.collect()

scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
scheduler_state = torch.load(
    scheduler_path, map_location="cpu", weights_only=True, mmap=True
)
require(isinstance(scheduler_state, dict), "scheduler.pt is not a mapping")
require(
    scheduler_state.get("last_epoch") == 10_000,
    f"scheduler last_epoch={scheduler_state.get('last_epoch')!r}, expected 10000",
)
base_lrs = scheduler_state.get("base_lrs")
last_lrs = scheduler_state.get("_last_lr")
require(
    isinstance(base_lrs, list) and isinstance(last_lrs, list)
    and len(base_lrs) == len(last_lrs) == len(optimizer_lrs),
    "scheduler LR layout does not match optimizer parameter groups",
)
for index, (scheduler_lr, optimizer_lr) in enumerate(zip(last_lrs, optimizer_lrs)):
    require(
        math.isfinite(float(scheduler_lr))
        and math.isclose(float(scheduler_lr), optimizer_lr, rel_tol=1e-12, abs_tol=0.0),
        f"scheduler LR {scheduler_lr!r} does not match optimizer group {index} LR {optimizer_lr!r}",
    )
for path, tensor in walk_tensors(scheduler_state):
    if tensor.is_floating_point() or tensor.is_complex():
        require(torch.isfinite(tensor).all().item(), f"non-finite scheduler tensor: {path}")
del scheduler_state

for rank in range(8):
    rng_path = os.path.join(checkpoint_dir, f"rng_state_{rank}.pth")
    rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
    require(isinstance(rng_state, dict) and rng_state, f"rank {rank} RNG state is invalid")
    require(isinstance(rng_state.get("python"), tuple),
            f"rank {rank} Python RNG state is invalid")
    require(isinstance(rng_state.get("numpy"), tuple),
            f"rank {rank} NumPy RNG state is invalid")
    require("cpu" in rng_state, f"rank {rank} RNG state has no CPU state")
    require("cuda" in rng_state, f"rank {rank} RNG state has no CUDA state")
    cpu_rng = rng_state["cpu"]
    require(
        isinstance(cpu_rng, torch.Tensor) and cpu_rng.dtype == torch.uint8
        and cpu_rng.ndim == 1 and cpu_rng.numel() > 0,
        f"rank {rank} CPU RNG tensor is invalid",
    )
    cuda_rng = rng_state["cuda"]
    cuda_tensors = list(cuda_rng) if isinstance(cuda_rng, (list, tuple)) else [cuda_rng]
    require(bool(cuda_tensors), f"rank {rank} CUDA RNG state is empty")
    require(
        all(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.uint8
            and tensor.ndim == 1 and tensor.numel() > 0
            for tensor in cuda_tensors),
        f"rank {rank} CUDA RNG tensor layout is invalid",
    )
    require(
        any(True for _ in walk_tensors(rng_state)),
        f"rank {rank} RNG state contains no tensors",
    )
    del rng_state

with open(log_path, encoding="utf-8", errors="replace") as handle:
    text = handle.read()
fatal_patterns = (
    r"CUDA out of memory",
    r"OutOfMemoryError",
    r"ChildFailedError",
    r"ProcessExitedException",
    r"Segmentation fault",
    r"Bus error",
    r"device-side assert",
    r"CUDA error:",
    r"\bXid\b",
    r"NCCL[^\r\n]*(?:unhandled|system error|remote process exited|watchdog|collective operation timeout)",
    r"(?:RuntimeError|ValueError|AssertionError|TypeError|KeyError|IndexError|MemoryError):",
)
for pattern in fatal_patterns:
    require(
        re.search(pattern, text, flags=re.IGNORECASE) is None,
        f"fatal training-log signature matched: {pattern}",
    )
tracebacks = text.count("Traceback (most recent call last):")
if tracebacks:
    require(tracebacks == 1, f"found {tracebacks} tracebacks, expected at most one")
    require(
        re.search(r"SignalException: Process \d+ got signal: 15", text) is not None,
        "the only traceback is not the expected controlled SIGTERM",
    )

print(f"global_step={global_step}")
print(f"finite_metric_rows={len(metrics)}")
print(f"final_metrics={metrics[-1]}")
print(f"model_parameter_tensors={model_tensor_count}")
print(f"model_parameter_count={model_parameter_count}")
print(f"optimizer_tensors={optimizer_tensor_count}")
print("model_config_sidecar_optimizer_scheduler_rng=OK")
print(f"controlled_sigterm_tracebacks={tracebacks}")
PY

grep -F 'DONE: shared_local_tied_g16 — STOPPED at step 10000' \
    "$TASK_EXPERIMENT_LOG"
grep -F 'All 1 experiments done' "$TASK_EXPERIMENT_LOG"
tail -12 "$TASK_EXPERIMENT_LOG"
du -sh "$TASK_OUTPUT"
echo 'TH2 SHARED LOCAL TIED G16 CHECKPOINT 10000 COMPLETE'

echo '=== require all eight B200 GPUs to be free immediately before burn launch ==='
query_compute_pairs
[[ "${#TASK_COMPUTE_PAIRS[@]}" -eq 0 ]] \
    || die "GPU compute processes are still active: ${TASK_COMPUTE_PAIRS[*]}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader \
    || die 'failed to query free-GPU state'
echo 'TH2 ALL 8 B200 GPUS FREE AFTER TRAINING'

echo '=== validate the burn environment ==='
"$TASK_BURN_PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: burn Python cannot access CUDA")
print(f"burn_torch=OK version={torch.__version__} visible_devices={torch.cuda.device_count()}")
PY

echo '=== launch one supervised burn worker per physical GPU UUID ==='
TASK_CHILD_PIDS=()
declare -A TASK_BURN_START_BY_PID=()
declare -A TASK_BURN_PID_BY_UUID=()
declare -A TASK_BURN_LOG_BY_PID=()
TASK_CLEANUP_ACTIVE=1

cleanup_burn_workers() {
    local original_status=$?
    local pid attempt
    local -a survivors=()
    trap - EXIT INT TERM
    if [[ "$TASK_CLEANUP_ACTIVE" -eq 1 ]]; then
        for pid in "${TASK_CHILD_PIDS[@]}"; do
            if same_proc "$pid" "${TASK_BURN_START_BY_PID[$pid]}"; then
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
        for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
            survivors=()
            for pid in "${TASK_CHILD_PIDS[@]}"; do
                same_proc "$pid" "${TASK_BURN_START_BY_PID[$pid]}" \
                    && survivors+=("$pid")
            done
            [[ "${#survivors[@]}" -eq 0 ]] && break
            sleep 1
        done
        for pid in "${survivors[@]}"; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        for attempt in 1 2 3 4 5; do
            survivors=()
            for pid in "${TASK_CHILD_PIDS[@]}"; do
                same_proc "$pid" "${TASK_BURN_START_BY_PID[$pid]}" \
                    && survivors+=("$pid")
            done
            [[ "${#survivors[@]}" -eq 0 ]] && break
            sleep 1
        done
        for pid in "${TASK_CHILD_PIDS[@]}"; do
            if ! same_proc "$pid" "${TASK_BURN_START_BY_PID[$pid]}"; then
                wait "$pid" 2>/dev/null || true
            fi
        done
        if [[ "${#survivors[@]}" -gt 0 ]]; then
            echo "WARNING: burn processes survived TERM/KILL: ${survivors[*]}"
        fi
    fi
    return "$original_status"
}
trap cleanup_burn_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for TASK_UUID in "${TASK_EXPECTED_GPU_UUIDS[@]}"; do
    TASK_GPU=${TASK_GPU_INDEX_BY_UUID[$TASK_UUID]}
    TASK_BURN_LOG="/tmp/project_gpu_burn_gpu${TASK_GPU}.log"
    env CUDA_VISIBLE_DEVICES="$TASK_UUID" \
        "$TASK_BURN_PYTHON" -u "$TASK_BURN_SCRIPT" \
        >"$TASK_BURN_LOG" 2>&1 &
    TASK_PID=$!
    if ! proc_snapshot "$TASK_PID"; then
        kill -TERM "$TASK_PID" 2>/dev/null || true
        wait "$TASK_PID" 2>/dev/null || true
        die "could not snapshot new burn process $TASK_PID"
    fi
    TASK_CHILD_PIDS+=("$TASK_PID")
    TASK_BURN_START_BY_PID[$TASK_PID]=$TASK_PROC_START_TICKS
    TASK_BURN_PID_BY_UUID[$TASK_UUID]=$TASK_PID
    TASK_BURN_LOG_BY_PID[$TASK_PID]=$TASK_BURN_LOG
    echo "launched gpu=$TASK_GPU uuid=$TASK_UUID pid=$TASK_PID start_ticks=$TASK_PROC_START_TICKS log=$TASK_BURN_LOG"
done

declare -A TASK_EXPECTED_BURN_PAIR=()
TASK_EXPECTED_BURN_PAIRS=$(for TASK_UUID in "${TASK_EXPECTED_GPU_UUIDS[@]}"; do
    printf '%s,%s\n' "$TASK_UUID" "${TASK_BURN_PID_BY_UUID[$TASK_UUID]}"
done | sort)
while IFS= read -r TASK_PAIR; do
    [[ -n "$TASK_PAIR" ]] && TASK_EXPECTED_BURN_PAIR[$TASK_PAIR]=1
done <<<"$TASK_EXPECTED_BURN_PAIRS"

burn_children_alive() {
    local pid
    for pid in "${TASK_CHILD_PIDS[@]}"; do
        if ! same_proc "$pid" "${TASK_BURN_START_BY_PID[$pid]}"; then
            echo "ERROR: burn child $pid exited or changed identity"
            tail -50 "${TASK_BURN_LOG_BY_PID[$pid]}" 2>/dev/null || true
            return 1
        fi
        read_proc_cmdline "$pid" || return 1
        case "$TASK_PROC_CMDLINE" in
            *scripts/gpu_burn.py*) ;;
            *) echo "ERROR: burn child $pid has unexpected argv: $TASK_PROC_CMDLINE"; return 1 ;;
        esac
    done
}

burn_mapping_exact() {
    local pair pid actual
    query_compute_pairs
    for pair in "${TASK_COMPUTE_PAIRS[@]}"; do
        [[ -n "${TASK_EXPECTED_BURN_PAIR[$pair]+x}" ]] \
            || die "unexpected GPU process during burn startup/supervision: $pair"
        pid=${pair#*,}
        same_proc "$pid" "${TASK_BURN_START_BY_PID[$pid]}" \
            || die "GPU burn PID $pid changed identity"
    done
    [[ "${#TASK_COMPUTE_PAIRS[@]}" -eq 8 ]] || return 1
    actual=$(printf '%s\n' "${TASK_COMPUTE_PAIRS[@]}" | sort)
    [[ "$actual" == "$TASK_EXPECTED_BURN_PAIRS" ]]
}

query_gpu_health() {
    local raw index uuid memory utilization
    raw=$(nvidia-smi \
        --query-gpu=index,uuid,memory.used,utilization.gpu \
        --format=csv,noheader,nounits) \
        || die 'failed to query GPU burn health'
    declare -gA TASK_GPU_MEMORY_BY_UUID=()
    declare -gA TASK_GPU_UTIL_BY_UUID=()
    TASK_GPU_HEALTH_LINES=()
    while IFS=, read -r index uuid memory utilization; do
        index=$(trim "${index:-}")
        uuid=$(trim "${uuid:-}")
        memory=$(trim "${memory:-}")
        utilization=$(trim "${utilization:-}")
        [[ "$index" =~ ^[0-9]+$ && "$memory" =~ ^[0-9]+$ \
            && "$utilization" =~ ^[0-9]+$ ]] \
            || die "invalid GPU health row: $index,$uuid,$memory,$utilization"
        [[ -n "${TASK_GPU_INDEX_BY_UUID[$uuid]+x}" ]] \
            || die "health query returned an unexpected GPU UUID: $uuid"
        TASK_GPU_MEMORY_BY_UUID[$uuid]=$memory
        TASK_GPU_UTIL_BY_UUID[$uuid]=$utilization
        TASK_GPU_HEALTH_LINES+=("gpu=$index uuid=$uuid memory_mib=$memory utilization_pct=$utilization")
    done <<<"$raw"
    [[ "${#TASK_GPU_HEALTH_LINES[@]}" -eq 8 ]] \
        || die "GPU health query returned ${#TASK_GPU_HEALTH_LINES[@]} rows"
}

echo '=== wait for all eight burns to become ready with exact UUID/PID mapping ==='
TASK_BURN_READY_DEADLINE=$((SECONDS + 180))
while true; do
    burn_children_alive || die 'a burn child failed during startup'
    TASK_ALL_READY=1
    for TASK_PID in "${TASK_CHILD_PIDS[@]}"; do
        grep -Fq 'gpu_burn_ready' "${TASK_BURN_LOG_BY_PID[$TASK_PID]}" \
            || TASK_ALL_READY=0
    done
    if burn_mapping_exact && [[ "$TASK_ALL_READY" -eq 1 ]]; then
        break
    fi
    (( SECONDS < TASK_BURN_READY_DEADLINE )) \
        || die 'burn workers did not become ready on all eight GPUs within 180 seconds'
    sleep 5
done

echo '=== verify every burn has substantial memory and active compute ==='
TASK_ACTIVITY_DEADLINE=$((SECONDS + 120))
while true; do
    query_gpu_health
    TASK_ALL_ACTIVE=1
    for TASK_UUID in "${TASK_EXPECTED_GPU_UUIDS[@]}"; do
        (( TASK_GPU_MEMORY_BY_UUID[$TASK_UUID] >= 1000 )) || TASK_ALL_ACTIVE=0
        (( TASK_GPU_UTIL_BY_UUID[$TASK_UUID] >= 50 )) || TASK_ALL_ACTIVE=0
    done
    [[ "$TASK_ALL_ACTIVE" -eq 1 ]] && break
    (( SECONDS < TASK_ACTIVITY_DEADLINE )) \
        || die 'all eight burn workers did not reach active utilization within 120 seconds'
    sleep 5
done
printf '%s\n' "${TASK_GPU_HEALTH_LINES[@]}"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader || die 'failed to print final burn process state'
echo 'TH2 GPU BURN VERIFIED: RUNNING ON ALL 8 B200 GPUS'

echo '=== continuously supervise exact UUID/PID mapping and burn activity ==='
declare -A TASK_LOW_UTIL_STREAK=()
TASK_HEALTH_ITERATION=0
for TASK_UUID in "${TASK_EXPECTED_GPU_UUIDS[@]}"; do
    TASK_LOW_UTIL_STREAK[$TASK_UUID]=0
done
while true; do
    sleep 15
    burn_children_alive || die 'a supervised burn child failed'
    burn_mapping_exact || die 'the exact eight-GPU burn mapping changed'
    query_gpu_health
    for TASK_UUID in "${TASK_EXPECTED_GPU_UUIDS[@]}"; do
        (( TASK_GPU_MEMORY_BY_UUID[$TASK_UUID] >= 1000 )) \
            || die "burn memory dropped below 1000 MiB on GPU $TASK_UUID"
        if (( TASK_GPU_UTIL_BY_UUID[$TASK_UUID] < 50 )); then
            TASK_LOW_UTIL_STREAK[$TASK_UUID]=$((TASK_LOW_UTIL_STREAK[$TASK_UUID] + 1))
        else
            TASK_LOW_UTIL_STREAK[$TASK_UUID]=0
        fi
        (( TASK_LOW_UTIL_STREAK[$TASK_UUID] < 4 )) \
            || die "burn utilization stayed below 50% for 60 seconds on GPU $TASK_UUID"
    done
    TASK_HEALTH_ITERATION=$((TASK_HEALTH_ITERATION + 1))
    if (( TASK_HEALTH_ITERATION % 4 == 0 )); then
        date -u
        printf '%s\n' "${TASK_GPU_HEALTH_LINES[@]}"
        echo 'TH2 GPU BURN SUPERVISOR HEALTHY'
    fi
done
