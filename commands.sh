#1 +300+a
#th2-train-raw-tiered-and-unified-ranklift-10k-20260906-a01
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

read_gpu_pids() {
    local destination_name="$1" output
    local -n destination="$destination_name"
    output="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" || return 1
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
    [[ "${#TASK_GPU_NAMES[@]}" -eq 8 ]] || die "expected 8 GPUs"
    for TASK_GPU_INDEX in "${!TASK_GPU_NAMES[@]}"; do
        [[ "${TASK_GPU_NAMES[$TASK_GPU_INDEX]}" == *B200* ]] \
            || die "GPU $TASK_GPU_INDEX is not B200: ${TASK_GPU_NAMES[$TASK_GPU_INDEX]}"
    done
}

require_free_gpus() {
    local stage="$1"
    TASK_STAGE_GPU_PIDS=()
    read_gpu_pids TASK_STAGE_GPU_PIDS || die "nvidia-smi query failed $stage"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv,noheader
    [[ "${#TASK_STAGE_GPU_PIDS[@]}" -eq 0 ]] \
        || die "GPU compute processes remain $stage: ${TASK_STAGE_GPU_PIDS[*]}"
    echo "ALL 8 B200 GPUS FREE $stage"
}

require_burn_ancestry() {
    local worker_pid="$1" current_pid="$1" parent_pid cmdline depth found=0
    for depth in 1 2 3 4 5 6; do
        [[ -r "/proc/$current_pid/status" ]] \
            || die "cannot inspect burn worker ancestry for PID $worker_pid"
        parent_pid="$(awk '/^PPid:/ {print $2}' "/proc/$current_pid/status")"
        [[ "$parent_pid" =~ ^[0-9]+$ ]] || die "invalid parent PID"
        [[ "$parent_pid" -ne 1 ]] || break
        [[ -r "/proc/$parent_pid/cmdline" ]] || die "cannot read parent $parent_pid"
        cmdline="$(tr '\0' ' ' < "/proc/$parent_pid/cmdline")"
        echo "burn_ancestry worker=$worker_pid depth=$depth parent=$parent_pid cmdline=$cmdline"
        if [[ "$cmdline" == *'/tmp/llm_pretrain_burn.py'* ]]; then
            found=1
            break
        fi
        current_pid="$parent_pid"
    done
    [[ "$found" -eq 1 ]] || die "GPU PID $worker_pid has no runner-burn ancestor"
}

verify_one_process_per_gpu() {
    local expected_kind="$1" gpu_index output
    local -a per_gpu_pids unique_pids
    per_gpu_pids=()
    for gpu_index in 0 1 2 3 4 5 6 7; do
        output="$(nvidia-smi -i "$gpu_index" --query-compute-apps=pid \
            --format=csv,noheader,nounits)" || die "failed to inspect GPU $gpu_index"
        mapfile -t TASK_ONE_GPU_PIDS < <(
            printf '%s\n' "$output" |
                sed 's/^[[:space:]]*//;s/[[:space:]]*$//;/^$/d' | sort -nu
        )
        [[ "${#TASK_ONE_GPU_PIDS[@]}" -eq 1 ]] \
            || die "GPU $gpu_index does not have exactly one $expected_kind process"
        [[ "${TASK_ONE_GPU_PIDS[0]}" -ne 1 ]] || die 'refusing PID 1'
        kill -0 "${TASK_ONE_GPU_PIDS[0]}" || die "$expected_kind process disappeared"
        per_gpu_pids+=("${TASK_ONE_GPU_PIDS[0]}")
        echo "gpu=$gpu_index kind=$expected_kind pid=${TASK_ONE_GPU_PIDS[0]}"
    done
    mapfile -t unique_pids < <(printf '%s\n' "${per_gpu_pids[@]}" | sort -nu)
    [[ "${#unique_pids[@]}" -eq 8 ]] || die "expected eight distinct $expected_kind processes"
}

start_burn_and_verify() {
    local burn_log="$1"
    test -s /tmp/llm_pretrain_burn.py || return 1
    TASK_PRE_BURN_PIDS=()
    read_gpu_pids TASK_PRE_BURN_PIDS || return 1
    [[ "${#TASK_PRE_BURN_PIDS[@]}" -eq 0 ]] || return 1
    mkdir -p "$(dirname "$burn_log")"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        python3 /tmp/llm_pretrain_burn.py >>"$burn_log" 2>&1 &
    echo "burn_launcher_pid=$!"
    sleep 30
    TASK_NEW_BURN_PIDS=()
    read_gpu_pids TASK_NEW_BURN_PIDS || return 1
    [[ "${#TASK_NEW_BURN_PIDS[@]}" -eq 8 ]] || return 1
    verify_one_process_per_gpu burn
    for TASK_PID in "${TASK_NEW_BURN_PIDS[@]}"; do require_burn_ancestry "$TASK_PID"; done
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
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
            echo 'ERROR: cannot query GPUs during exit; burn not started' >&2
        elif [[ "${#TASK_EXIT_GPU_PIDS[@]}" -eq 0 ]]; then
            echo "=== restoring runner burn during exit rc=$rc ==="
            start_burn_and_verify "$TASK_BURN_LOG" \
                || echo 'ERROR: could not restore runner burn during exit' >&2
        else
            echo "BURN NOT STARTED: GPU processes remain: ${TASK_EXIT_GPU_PIDS[*]}" >&2
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
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/raw_tiered_unified_ranklift_10k_20260906_a01"
TASK_STATUS_DIR="$TASK_OUTPUT_BASE/status"
TASK_COMPLETION_FILE="$TASK_STATUS_DIR/raw_tiered_unified_ranklift_10k_20260906_a01.complete"
TASK_COMPLETION_TMP="$TASK_COMPLETION_FILE.tmp"
TASK_BURN_LOG="$TASK_OUTPUT_BASE/logs/gpu_burn_after_raw_tiered_unified_ranklift_20260906_a01.log"
TASK_ACCELERATE_SOURCE="$TASK_PROJECT_DIR/resources/accelerate_config.yaml"
TASK_ACCELERATE_TARGET=/mnt/local/.cache/huggingface/accelerate/default_config.yaml
TASK_OUTPUT_DIRS=(
    "$TASK_OUTPUT_BASE/tiered_ranklift_raw_t4_c512"
    "$TASK_OUTPUT_BASE/unified_ranklift_raw_t4_m460"
)
TASK_LAUNCHERS=(
    scripts/train_tiered_ranklift_raw_tied.sh
    scripts/train_unified_ranklift_raw_tied.sh
)
TASK_NEEDS_BURN_RESTORE=0
trap restore_burn_on_exit EXIT

echo '=== identity and static preflight ==='
date -u
hostname
cd "$TASK_PROJECT_DIR"
test -x "$TASK_CONDA" && test -x "$TASK_PYTHON"
test -s "$TASK_MODEL_DIR/config.json" && test -s "$TASK_MODEL_DIR/tokenizer.json"
test -d "$TASK_DATA_DIR" && test -s "$TASK_ACCELERATE_SOURCE"
test -s /tmp/llm_pretrain_burn.py
echo 'c2694ea0b34f69119472a4262013b2387166aaa1981c80ac5b30c32b72c5741c  resources/token_freq_sample10.npz' | sha256sum -c -
echo '923db7f20a2df3d051180f67f9bea1f30c84c804651e313fa9961a9fd17a57e5  resources/accelerate_config.yaml' | sha256sum -c -
echo '417560b59cefdff13f7bface7b261eeba16edd0b428be40bb63c575aa2b4971c  compositional/unified_ranklift.py' | sha256sum -c -
echo '0000da269c8ae8378cf745bd9aac4f473f5a49574a377e8bf693dcca9a673305  scripts/train_tiered_ranklift_raw_tied.sh' | sha256sum -c -
echo '1ab121737c4b684a61c95a1ee1e8bf2b6fcd30d852134eb0eb14b051005c14a8  scripts/train_unified_ranklift_raw_tied.sh' | sha256sum -c -

for TASK_LAUNCHER in "${TASK_LAUNCHERS[@]}"; do
    bash -n "$TASK_LAUNCHER"
    grep -Eq '^[[:space:]]*--bf16([[:space:]]|$)' "$TASK_LAUNCHER"
    grep -Eq '^[[:space:]]*--ddp_find_unused_parameters[[:space:]]+false([[:space:]]|$)' "$TASK_LAUNCHER"
    grep -Eq '^[[:space:]]*--per_device_train_batch_size[[:space:]]+16([[:space:]]|$)' "$TASK_LAUNCHER"
    grep -Eq '^[[:space:]]*--gradient_accumulation_steps[[:space:]]+4([[:space:]]|$)' "$TASK_LAUNCHER"
    grep -Eq '^[[:space:]]*--save_steps[[:space:]]+250([[:space:]]|$)' "$TASK_LAUNCHER"
    grep -Eq '^[[:space:]]*--tie_output([[:space:]]|$)' "$TASK_LAUNCHER"
    ! grep -Eq '^[[:space:]]*--max_steps([[:space:]]|$)' "$TASK_LAUNCHER"
done
grep -Fq -- '--tiered_ranklift_code_dims 1024,512,192,64' scripts/train_tiered_ranklift_raw_tied.sh
grep -Fq -- '--tiered_ranklift_lift_dims 0,0,320,192' scripts/train_tiered_ranklift_raw_tied.sh
grep -Fq -- '--unified_ranklift_code_dims 384,256,128,112' scripts/train_unified_ranklift_raw_tied.sh
grep -Fq -- '--unified_ranklift_feature_dim 460' scripts/train_unified_ranklift_raw_tied.sh
for TASK_LANG in en vi zh ru de ar; do
    test -d "$TASK_DATA_DIR/$TASK_LANG" || die "missing training language: $TASK_LANG"
    find "$TASK_DATA_DIR/$TASK_LANG" -type f -name '*.arrow' ! -name 'cache-*' \
        -print -quit | grep -q . || die "no source Arrow file for $TASK_LANG"
done
for TASK_OUTPUT_DIR in "${TASK_OUTPUT_DIRS[@]}"; do
    [[ ! -e "$TASK_OUTPUT_DIR" ]] || die "refusing stale output: $TASK_OUTPUT_DIR"
done
[[ ! -e "$TASK_LOG_DIR" ]] || die "refusing stale log directory: $TASK_LOG_DIR"
[[ ! -e "$TASK_COMPLETION_FILE" && ! -e "$TASK_COMPLETION_TMP" ]] \
    || die "refusing stale completion marker"

echo '=== activate and verify sparse_emb ==='
eval "$("$TASK_CONDA" shell.bash hook)"
conda activate sparse_emb
[[ "${CONDA_DEFAULT_ENV:-}" == sparse_emb ]]
[[ "$(command -v python3.11)" == "$TASK_PYTHON" ]]
"$TASK_PYTHON" - "$TASK_MODEL_DIR/config.json" <<'PY'
import json, sys
import accelerate, datasets, torch, transformers
with open(sys.argv[1], encoding="utf-8") as handle: config = json.load(handle)
assert config["model_type"] == "qwen3"
assert config["hidden_size"] == 1024 and config["vocab_size"] == 151936
assert torch.cuda.is_available() and torch.cuda.device_count() == 8
assert all("B200" in torch.cuda.get_device_name(i) for i in range(8))
print("TRAINING_RUNTIME_OK", torch.__version__, transformers.__version__,
      datasets.__version__, accelerate.__version__)
PY

"$TASK_PYTHON" - <<'PY'
from train_compositional import CompositionalArguments, build_arm
configs = (
 ("tiered_ranklift_raw_t4_c512", CompositionalArguments(
   arm="tiered_ranklift", tie_output=True,
   tiered_ranklift_code_dims="1024,512,192,64",
   tiered_ranklift_lift_dims="0,0,320,192",
   tiered_ranklift_populations="2048,6144,24576,119168",
   tiered_ranklift_frequency_path="resources/token_freq_sample10.npz"), 20_096_000),
 ("unified_ranklift_raw_t4_m460", CompositionalArguments(
   arm="unified_ranklift", tie_output=True,
   unified_ranklift_code_dims="384,256,128,112", unified_ranklift_feature_dim=460,
   unified_ranklift_populations="2048,6144,24576,119168",
   unified_ranklift_frequency_path="resources/token_freq_sample10.npz"), 19_651_584),
)
for name, config, expected in configs:
    module = build_arm(config, 151_936, 1_024)
    count = sum(p.numel() for p in module.parameters())
    assert count == expected and module.group_sizes == (2048,6144,24576,119168)
    print(f"INTERFACE_CONFIG_OK name={name} params={count}")
PY
require_b200_node

echo '=== verify burn ownership, then stop only its eight GPU workers ==='
if pgrep -af '[r]un_experiments.py|[t]rain_compositional.py|[e]val_parallel.py|[f]inetune/run_all.py|[f]inetune/train.py|[p]repare_data.py'; then
    die 'a real project workload exists'
fi
TASK_BURN_PIDS=()
read_gpu_pids TASK_BURN_PIDS || die 'failed to query current GPU PIDs'
[[ "${#TASK_BURN_PIDS[@]}" -eq 8 ]] || die "expected 8 burn workers, found ${#TASK_BURN_PIDS[@]}"
verify_one_process_per_gpu burn
for TASK_PID in "${TASK_BURN_PIDS[@]}"; do require_burn_ancestry "$TASK_PID"; done
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
TASK_NEEDS_BURN_RESTORE=1
kill -9 "${TASK_BURN_PIDS[@]}" 2>/dev/null || true
sleep 30
require_free_gpus '30 SECONDS AFTER BURN STOP'

echo '=== copy Accelerate config, wait, and recheck free GPUs ==='
mkdir -p "$(dirname "$TASK_ACCELERATE_TARGET")"
cp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
cmp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
grep -Fxq 'distributed_type: MULTI_GPU' "$TASK_ACCELERATE_TARGET"
grep -Fxq 'mixed_precision: bf16' "$TASK_ACCELERATE_TARGET"
grep -Fxq 'num_processes: 8' "$TASK_ACCELERATE_TARGET"
sleep 30
require_free_gpus '30 SECONDS AFTER ACCELERATE CONFIG COPY'

export SPARSE_EMB_PYTHON="$TASK_PYTHON"
export SPARSE_EMB_MODEL_DIR="$TASK_MODEL_DIR"
export SPARSE_EMB_DATA_DIR="$TASK_DATA_DIR"
export SPARSE_EMB_OUTPUT_BASE="$TASK_OUTPUT_BASE"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset BTMOS_OUTPUT_DIR BTMOS_IMPORTANCE_PATH || true

mapfile -t TASK_EXPERIMENT_INDICES < <(
"$TASK_PYTHON" - <<'PY'
from run_experiments import EXPERIMENT_COMMANDS
expected = (
 ("tiered_ranklift_raw_t4_c512", "bash scripts/train_tiered_ranklift_raw_tied.sh"),
 ("unified_ranklift_raw_t4_m460", "bash scripts/train_unified_ranklift_raw_tied.sh"),
)
for name, command in expected:
    matches=[(i,e) for i,e in enumerate(EXPERIMENT_COMMANDS) if e.get("name")==name]
    assert len(matches)==1
    index, experiment=matches[0]
    assert experiment["cmd"]==command and experiment.get("require_fresh_output") is True
    print(index)
PY
)
[[ "${#TASK_EXPERIMENT_INDICES[@]}" -eq 2 ]]
printf 'selected_experiment_indices=%s %s\n' "${TASK_EXPERIMENT_INDICES[@]}"
echo 'order=tiered_ranklift_raw_t4_c512 then unified_ranklift_raw_t4_m460'
echo 'bf16, seed 42, effective batch 512, one-epoch schedule, stop-at-step 10000'
mkdir -p "$TASK_LOG_DIR"
"$TASK_PYTHON" -u run_experiments.py \
    --experiments "${TASK_EXPERIMENT_INDICES[@]}" \
    --stop-at-step 10000 --log-dir "$TASK_LOG_DIR"

sleep 30
require_free_gpus '30 SECONDS AFTER BOTH TRAINING RUNS'
"$TASK_PYTHON" - "$TASK_LOG_DIR" "${TASK_OUTPUT_DIRS[@]}" <<'PY'
import json, math, pathlib, sys
log_dir=pathlib.Path(sys.argv[1]); output_dirs=[pathlib.Path(x) for x in sys.argv[2:]]
required=("config.json","model.safetensors","trainer_state.json","optimizer.pt",
 "scheduler.pt","embedding.pt","rng_state_0.pth","rng_state_1.pth",
 "rng_state_2.pth","rng_state_3.pth","rng_state_4.pth","rng_state_5.pth",
 "rng_state_6.pth","rng_state_7.pth")
fatal=("CUDA out of memory","OutOfMemoryError","ChildFailedError",
       "ProcessExitedException","Segmentation fault","Bus error")
for output_dir in output_dirs:
    checkpoint=output_dir/"checkpoint-10000"
    missing=[x for x in required if not (checkpoint/x).is_file() or not (checkpoint/x).stat().st_size]
    assert not missing,(checkpoint,missing)
    state=json.loads((checkpoint/"trainer_state.json").read_text())
    assert int(state["global_step"])==10000
    losses=[float(x["loss"]) for x in state.get("log_history",[]) if "loss" in x]
    assert losses and all(math.isfinite(x) for x in losses)
    cfg=json.loads((output_dir/"train_config.json").read_text())
    comp=cfg["compositional"]; training=cfg["training"]
    assert comp["tie_output"] is True and training["ddp_find_unused_parameters"] is False
    if output_dir.name=="tiered_ranklift_raw_t4_c512":
        assert comp["arm"]=="tiered_ranklift"
        assert comp["tiered_ranklift_code_dims"]=="1024,512,192,64"
        assert comp["tiered_ranklift_lift_dims"]=="0,0,320,192"
        assert comp["tiered_ranklift_frequency_path"]=="resources/token_freq_sample10.npz"
    else:
        assert comp["arm"]=="unified_ranklift"
        assert comp["unified_ranklift_code_dims"]=="384,256,128,112"
        assert int(comp["unified_ranklift_feature_dim"])==460
        assert comp["unified_ranklift_frequency_path"]=="resources/token_freq_sample10.npz"
    text=(log_dir/f"{output_dir.name}.log").read_text(errors="replace")
    found=[x for x in fatal if x in text]; assert not found,(output_dir.name,found)
    print(f"CHECKPOINT_10000_VALID name={output_dir.name} first={losses[0]:.6f} last={losses[-1]:.6f}")
summary=(log_dir/"experiments.log").read_text(errors="replace")
for output_dir in output_dirs:
    assert f"{output_dir.name}: STOPPED at step 10000" in summary
print("BOTH_EXPERIMENT_LOG_STATUSES_VALID")
PY

mkdir -p "$TASK_STATUS_DIR"
{
    printf 'STATUS=SUCCESS\nCOMPLETED_UTC=%s\nSTOP_STEP=10000\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf 'EXPERIMENT_1=tiered_ranklift_raw_t4_c512\n'
    printf 'EXPERIMENT_2=unified_ranklift_raw_t4_m460\n'
    printf 'CHECKPOINT_1=%s\nCHECKPOINT_2=%s\nLOG_DIR=%s\n' \
        "${TASK_OUTPUT_DIRS[0]}/checkpoint-10000" \
        "${TASK_OUTPUT_DIRS[1]}/checkpoint-10000" "$TASK_LOG_DIR"
} >"$TASK_COMPLETION_TMP"
mv "$TASK_COMPLETION_TMP" "$TASK_COMPLETION_FILE"
cat "$TASK_COMPLETION_FILE"

start_burn_and_verify "$TASK_BURN_LOG"
TASK_NEEDS_BURN_RESTORE=0
echo 'TH2 RAW TIERED AND UNIFIED RANKLIFT TRAINING COMPLETE; BURNS ACTIVE'
