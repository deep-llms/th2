#1 +240+a
#th2-wait-hashed-then-retry-rse-b8a8-10k-and-burn-20260831-a01
set -euo pipefail

TASK_PROJECT=/mnt/local/@PROJECT@
TASK_CONDA=/mnt/local/conda-py311/bin/conda
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_MODEL=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_DATA=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/train
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_HASHED_OUTPUT="$TASK_OUTPUT_BASE/product_code_hashed_h2048"
TASK_RSE_OUTPUT="$TASK_OUTPUT_BASE/residual_subspace_experts_tied_g12_r120_q80"
TASK_RETRY_LOG_DIR="$TASK_OUTPUT_BASE/logs/final_rse_retry_b8a8_10k_20260831_a01"
TASK_ACCELERATE_SOURCE="$TASK_PROJECT/resources/accelerate_config.yaml"
TASK_ACCELERATE_DEST=/mnt/local/.cache/huggingface/accelerate/default_config.yaml
TASK_BURN=/tmp/llm_pretrain_burn.py
TASK_BURN_SHA=2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174
TASK_HANDOFF_BURN_LOG=/tmp/llm_pretrain_burn_after_final_rse_hashed_10k.log
TASK_RETRY_BURN_LOG=/tmp/llm_pretrain_burn_after_final_rse_retry_b8a8_10k.log
TASK_BURN_PID_FILE=/tmp/llm_pretrain_burn_launcher.pid

gpu_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d;s/[[:space:]]//g' | sort -nu
}

assert_gpus_free() {
  local -a pids
  mapfile -t pids < <(gpu_pids)
  test "${#pids[@]}" -eq 0 || {
    echo "ERROR: expected free GPUs; found PIDs ${pids[*]}" >&2
    nvidia-smi
    return 1
  }
  nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
}

verify_burn() {
  local log_path="$1"
  local -n workers_ref="$2"
  local launcher='' pid ppid launcher_cmd marker count

  echo "$TASK_BURN_SHA  $TASK_BURN" | sha256sum -c -
  test -s "$log_path"
  mapfile -t workers_ref < <(gpu_pids)
  test "${#workers_ref[@]}" -eq 8
  for pid in "${workers_ref[@]}"; do
    test "$pid" != 1
    ppid="$(awk '/^PPid:/ {print $2}' "/proc/$pid/status")"
    test "$ppid" != 1
    if [[ -z "$launcher" ]]; then launcher="$ppid"; else test "$ppid" = "$launcher"; fi
  done
  launcher_cmd="$(tr '\0' ' ' <"/proc/$launcher/cmdline")"
  [[ "$launcher_cmd" == *"$TASK_BURN"* ]]
  for marker in \
      gpu_burn_ready world_size=8 collective_probe_sum=36 \
      comm_total_mib=1137 comm_bucket_mib=25 approx_step_seconds=0.750; do
    count="$(grep -oF "$marker" "$log_path" | wc -l || true)"
    test "$count" -eq 8
  done
  grep -Fq 'gpu_burn_progress' "$log_path"
  echo "verified_burn_log=$log_path launcher=$launcher workers=${workers_ref[*]}"
}

cd "$TASK_PROJECT"
echo '=== retry watcher immutable preflight; do not touch the live Hashed run ==='
date -u
hostname
pwd
echo '26dfd72c83b3c23aad9d21e13f51b7ce1f8914b9a3a4c325de5bf371cda91529  scripts/train_residual_subspace_experts_tied.sh' | sha256sum -c -
echo '726a15849995136c313a2e4c888988385216387b4ac3f1f5a57e61f6ea86e171  run_experiments.py' | sha256sum -c -
echo '58a4dfdbe46d45ce674a1f6d3cd8501941b0e8eccda2dc1a047eaeda3fa4bf35  train_compositional.py' | sha256sum -c -
echo 'f2f5ea64b63d933d7e0609bd27f3226bf4c8abcf7bcf582a5907557ecb2ab606  compositional/residual_subspace_experts.py' | sha256sum -c -
echo '923db7f20a2df3d051180f67f9bea1f30c84c804651e313fa9961a9fd17a57e5  resources/accelerate_config.yaml' | sha256sum -c -
echo "$TASK_BURN_SHA  $TASK_BURN" | sha256sum -c -
test -x "$TASK_CONDA"
test -x "$TASK_PYTHON"
test -d "$TASK_MODEL"
test -s "$TASK_MODEL/config.json"
test -d "$TASK_DATA"
test -d "$TASK_HASHED_OUTPUT"
test -d "$TASK_RSE_OUTPUT"
test ! -e "$TASK_RSE_OUTPUT/checkpoint-10000"
test ! -e "$TASK_RETRY_LOG_DIR"

echo '=== wait for Hashed checkpoint-10000 and the first handoff burn restoration ==='
TASK_READY=0
for TASK_WAIT_INDEX in $(seq 1 1000); do
  if [[ -s "$TASK_HASHED_OUTPUT/checkpoint-10000/trainer_state.json" ]] && \
     [[ -s "$TASK_HANDOFF_BURN_LOG" ]] && \
     grep -Fq 'gpu_burn_progress' "$TASK_HANDOFF_BURN_LOG"; then
    TASK_READY=1
    break
  fi
  if (( TASK_WAIT_INDEX % 10 == 1 )); then
    echo "waiting_for_hashed_and_handoff_burn poll=$TASK_WAIT_INDEX utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  fi
  sleep 60
done
test "$TASK_READY" -eq 1

echo '=== validate completed Hashed checkpoint before reclaiming the restored burn ==='
for TASK_REQUIRED in config.json model.safetensors trainer_state.json optimizer.pt scheduler.pt \
    embedding.pt rng_state_0.pth rng_state_1.pth rng_state_2.pth rng_state_3.pth \
    rng_state_4.pth rng_state_5.pth rng_state_6.pth rng_state_7.pth; do
  test -s "$TASK_HASHED_OUTPUT/checkpoint-10000/$TASK_REQUIRED"
done
"$TASK_PYTHON" - "$TASK_HASHED_OUTPUT/checkpoint-10000/trainer_state.json" <<'PY'
import json
import math
import sys

state = json.load(open(sys.argv[1]))
assert state["global_step"] == 10000
losses = [row for row in state["log_history"] if "loss" in row]
assert losses
assert all(math.isfinite(float(row["loss"])) for row in losses)
print(f"HASHED_CHECKPOINT_10000_READY last_loss={losses[-1]}")
PY

mapfile -t TASK_HANDOFF_BURN_PIDS
verify_burn "$TASK_HANDOFF_BURN_LOG" TASK_HANDOFF_BURN_PIDS
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader

echo '=== stop only the verified post-Hashed burn workers ==='
kill -9 "${TASK_HANDOFF_BURN_PIDS[@]}" 2>/dev/null || true

set +e
(
  set -euo pipefail
  sleep 30
  assert_gpus_free
  echo 'ALL_EIGHT_GPUS_FREE_BEFORE_RSE_RETRY_CLEANUP'

  echo '=== remove only the failed pre-step RSE output, then recreate it fresh ==='
  find "$TASK_RSE_OUTPUT" -maxdepth 2 -mindepth 1 -printf '%P\n' | sort
  test ! -e "$TASK_RSE_OUTPUT/checkpoint-10000"
  rm -rf -- "$TASK_RSE_OUTPUT"
  test ! -e "$TASK_RSE_OUTPUT"

  eval "$("$TASK_CONDA" shell.bash hook)"
  conda activate sparse_emb
  test "$CONDA_DEFAULT_ENV" = sparse_emb
  test "$(command -v python3.11)" = "$TASK_PYTHON"
  mkdir -p "$(dirname "$TASK_ACCELERATE_DEST")"
  cp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_DEST"
  cmp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_DEST"
  grep -Fx 'distributed_type: MULTI_GPU' "$TASK_ACCELERATE_DEST"
  grep -Fx 'mixed_precision: bf16' "$TASK_ACCELERATE_DEST"
  grep -Fx 'num_processes: 8' "$TASK_ACCELERATE_DEST"

  sleep 30
  assert_gpus_free
  echo 'ALL_EIGHT_GPUS_FREE_BEFORE_RSE_B8A8_RETRY'

  export SPARSE_EMB_PYTHON="$TASK_PYTHON"
  export SPARSE_EMB_MODEL_DIR="$TASK_MODEL"
  export SPARSE_EMB_DATA_DIR="$TASK_DATA"
  export SPARSE_EMB_OUTPUT_BASE="$TASK_OUTPUT_BASE"
  export RSE_PER_DEVICE_TRAIN_BATCH_SIZE=8
  export RSE_GRADIENT_ACCUMULATION_STEPS=8
  export WANDB_MODE=offline
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  export NCCL_NVLS_ENABLE=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

  "$TASK_PYTHON" - <<'PY'
from run_experiments import EXPERIMENT_COMMANDS
experiment = EXPERIMENT_COMMANDS[23]
assert experiment["name"] == "residual_subspace_experts_tied_g12_r120_q80"
assert experiment["require_fresh_output"] is True
print(f"retry_experiment=23:{experiment['name']} cmd={experiment['cmd']}")
print("retry_batch=8 accumulation=8 world=8 global_batch=512")
PY

  mkdir -p "$TASK_RETRY_LOG_DIR"
  set +e
  "$TASK_PYTHON" run_experiments.py \
    --experiments 23 \
    --stop-at-step 10000 \
    --log-dir "$TASK_RETRY_LOG_DIR"
  TASK_RETRY_RC=$?
  set -e
  echo "rse_retry_runner_exit_code=$TASK_RETRY_RC"
  sleep 30

  test -s "$TASK_RETRY_LOG_DIR/experiments.log"
  test -s "$TASK_RETRY_LOG_DIR/residual_subspace_experts_tied_g12_r120_q80.log"
  grep -F 'residual_subspace_experts_tied_g12_r120_q80: STOPPED at step 10000' \
    "$TASK_RETRY_LOG_DIR/experiments.log"
  if grep -E -i 'Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError:|NCCL[^[:cntrl:]]*(unhandled|system error|remote process exited|watchdog|collective operation timeout)' \
      "$TASK_RETRY_LOG_DIR/residual_subspace_experts_tied_g12_r120_q80.log"; then
    echo 'ERROR: fatal signature found in RSE retry log' >&2
    exit 1
  fi

  TASK_CHECKPOINT="$TASK_RSE_OUTPUT/checkpoint-10000"
  test -s "$TASK_RSE_OUTPUT/train_config.json"
  test ! -e "$TASK_RSE_OUTPUT/output_head.pt"
  test ! -e "$TASK_CHECKPOINT/output_head.pt"
  for TASK_REQUIRED in config.json model.safetensors trainer_state.json optimizer.pt scheduler.pt \
      embedding.pt rng_state_0.pth rng_state_1.pth rng_state_2.pth rng_state_3.pth \
      rng_state_4.pth rng_state_5.pth rng_state_6.pth rng_state_7.pth; do
    test -s "$TASK_CHECKPOINT/$TASK_REQUIRED"
  done

  "$TASK_PYTHON" - "$TASK_RSE_OUTPUT" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch
from compositional.loading import _build_arm_from_config
from compositional.residual_subspace_experts import ResidualSubspaceExpertsEmbed

output = Path(sys.argv[1])
checkpoint = output / "checkpoint-10000"
trainer = json.loads((checkpoint / "trainer_state.json").read_text())
assert trainer["global_step"] == 10000
losses = [row for row in trainer["log_history"] if "loss" in row]
assert losses
for row in losses:
    for key in ("loss", "grad_norm", "learning_rate", "div_loss"):
        assert math.isfinite(float(row[key])), (key, row[key])
saved = json.loads((output / "train_config.json").read_text())
training = saved["training"]
assert training["bf16"] is True
assert training["per_device_train_batch_size"] == 8
assert training["gradient_accumulation_steps"] == 8
assert training["max_steps"] == -1
assert training["num_train_epochs"] == 1
assert training["seed"] == 42
assert training["learning_rate"] == 3e-4
assert saved["data"]["block_size"] == 2048
config = saved["compositional"]
assert config["arm"] == "residual_subspace_experts"
assert config["tie_output"] is True
assert config["independent_lowrank_output"] is False
assert config["rse_base_rank"] == 120
assert config["rse_expert_rank"] == 80
assert config["rse_num_experts"] == 12
assert config["rse_router_dim"] == 32
assert config["rse_top_k"] == 2
assert config["lambda_div"] == 0.01
state = torch.load(checkpoint / "embedding.pt", map_location="cpu", weights_only=True)
structure = ResidualSubspaceExpertsEmbed.structure_from_state(state)
assert structure["vocab_size"] == 151936 and structure["embed_dim"] == 1024
module = _build_arm_from_config(config, 151936, 1024, state=state)
module.load_state_dict(state, strict=True)
assert module.parameter_count == 19_471_968
assert state["expert_down_weight"].abs().max().item() > 0.0
assert state["expert_down_bias"].abs().max().item() > 0.0
for name, tensor in state.items():
    if tensor.is_floating_point():
        assert torch.isfinite(tensor).all(), name
print(f"RSE_B8A8_CHECKPOINT_10000_PASS last_loss={losses[-1]}")
PY
  test "$TASK_RETRY_RC" -eq 0
  assert_gpus_free
  echo 'ALL_EIGHT_GPUS_FREE_AFTER_RSE_RETRY'
)
TASK_WORK_RC=$?
set -e
echo "rse_retry_and_audit_exit_code=$TASK_WORK_RC"

echo '=== wait for any owned training workers to exit before burn restoration ==='
for TASK_WAIT_INDEX in $(seq 1 12); do
  mapfile -t TASK_AFTER_RETRY_PIDS < <(gpu_pids)
  (( ${#TASK_AFTER_RETRY_PIDS[@]} == 0 )) && break
  echo "waiting_for_retry_gpu_pids=${TASK_AFTER_RETRY_PIDS[*]}"
  sleep 5
done
if (( ${#TASK_AFTER_RETRY_PIDS[@]} )); then
  echo "ERROR: refusing to start burn while GPU PIDs remain: ${TASK_AFTER_RETRY_PIDS[*]}" >&2
  nvidia-smi
  exit 1
fi
assert_gpus_free

echo '=== restart and verify the correct communicating burn on all eight GPUs ==='
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  nohup /usr/bin/python3 "$TASK_BURN" >"$TASK_RETRY_BURN_LOG" 2>&1 &
TASK_NEW_BURN_LAUNCHER=$!
echo "$TASK_NEW_BURN_LAUNCHER" >"$TASK_BURN_PID_FILE"
sleep 30
mapfile -t TASK_RETRY_BURN_PIDS
verify_burn "$TASK_RETRY_BURN_LOG" TASK_RETRY_BURN_PIDS
for TASK_PID in "${TASK_RETRY_BURN_PIDS[@]}"; do
  TASK_PPID="$(awk '/^PPid:/ {print $2}' "/proc/$TASK_PID/status")"
  test "$TASK_PPID" = "$TASK_NEW_BURN_LAUNCHER"
done
TASK_GPU_STATE="$(
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits
)"
printf '%s\n' "$TASK_GPU_STATE"
"$TASK_PYTHON" - "$TASK_GPU_STATE" <<'PY'
import sys
rows = [[int(field.strip()) for field in line.split(",")[:4]]
        for line in sys.argv[1].splitlines()]
assert len(rows) == 8 and {row[0] for row in rows} == set(range(8))
assert all(140_000 <= used < total for _, used, total, _ in rows)
print("ALL_EIGHT_RESTORED_BURNS_USE_MOST_GPU_MEMORY")
PY
echo 'CORRECT_COMMUNICATING_BURN_RUNNING_ON_ALL_EIGHT_GPUS_AFTER_RSE_RETRY'
test "$TASK_WORK_RC" -eq 0
echo 'TH2 HASHED AND MEMORY_SAFE RSE 10K TRAINING COMPLETE'
