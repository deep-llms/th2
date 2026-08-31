#1 +240+a
#th2-train-final-rse-then-hashed-10k-20260831-a01
set -euo pipefail

TASK_PROJECT=/mnt/local/@PROJECT@
TASK_CONDA=/mnt/local/conda-py311/bin/conda
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_MODEL=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_DATA=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/train
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_RSE_OUTPUT="$TASK_OUTPUT_BASE/residual_subspace_experts_tied_g12_r120_q80"
TASK_HASHED_OUTPUT="$TASK_OUTPUT_BASE/product_code_hashed_h2048"
TASK_IMPORTANCE="$TASK_OUTPUT_BASE/product_code_smoke_artifacts/token_importance_langbalanced.npz"
TASK_IMPORTANCE_SHA=39b15eab8cf213d563dcf5137bb982e836bb8e3beba8e7def8dcddf21fe43594
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/final_rse_hashed_10k_20260831_a01"
TASK_ACCELERATE_SOURCE="$TASK_PROJECT/resources/accelerate_config.yaml"
TASK_ACCELERATE_DEST=/mnt/local/.cache/huggingface/accelerate/default_config.yaml
TASK_BURN=/tmp/llm_pretrain_burn.py
TASK_BURN_SHA=2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174
TASK_CURRENT_BURN_LOG=/tmp/llm_pretrain_burn_after_rse_smoke_a04.log
TASK_NEW_BURN_LOG=/tmp/llm_pretrain_burn_after_final_rse_hashed_10k.log
TASK_BURN_PID_FILE=/tmp/llm_pretrain_burn_launcher.pid

gpu_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d;s/[[:space:]]//g' | sort -nu
}

assert_gpus_free() {
  local -a pids
  mapfile -t pids < <(gpu_pids)
  if (( ${#pids[@]} )); then
    echo "ERROR: expected all GPUs free; found PIDs: ${pids[*]}" >&2
    for pid in "${pids[@]}"; do
      printf 'pid=%s cmd=' "$pid" >&2
      tr '\0' ' ' <"/proc/$pid/cmdline" >&2 2>/dev/null || true
      echo >&2
    done
    nvidia-smi
    return 1
  fi
  nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
}

cd "$TASK_PROJECT"

echo '=== immutable preflight before touching GPUs ==='
date -u
hostname
pwd
git rev-parse HEAD 2>/dev/null || true
echo '726a15849995136c313a2e4c888988385216387b4ac3f1f5a57e61f6ea86e171  run_experiments.py' | sha256sum -c -
echo '58a4dfdbe46d45ce674a1f6d3cd8501941b0e8eccda2dc1a047eaeda3fa4bf35  train_compositional.py' | sha256sum -c -
echo '7be9957197d39b454cfcdda08989588e545b6487c9b2302362e96afcdcd581b9  scripts/train_residual_subspace_experts_tied.sh' | sha256sum -c -
echo 'f4d4963c0fadc578933e429fdf436812aa8f208826d66da9ee51e95a89cace2d  scripts/train_product_code_tied.sh' | sha256sum -c -
echo 'f2f5ea64b63d933d7e0609bd27f3226bf4c8abcf7bcf582a5907557ecb2ab606  compositional/residual_subspace_experts.py' | sha256sum -c -
echo 'b17de9f5ab2211c0825922bfaf8f23ed9ecf98c7cae29cc3973074c5dac843a1  compositional/product_code.py' | sha256sum -c -
echo '71dd5bf67ceb0284ec4a3e677f4ed3f615371b0eb135864f2f079992dd623d2f  compositional/loading.py' | sha256sum -c -
echo 'bbfe3a7e542eaa2862654c5c83ee293e94ac33f346518be8a8a751dfd70b221e  compositional/tied_head.py' | sha256sum -c -
echo '923db7f20a2df3d051180f67f9bea1f30c84c804651e313fa9961a9fd17a57e5  resources/accelerate_config.yaml' | sha256sum -c -
test -x "$TASK_CONDA"
test -x "$TASK_PYTHON"
test -d "$TASK_MODEL"
test -s "$TASK_MODEL/config.json"
test -d "$TASK_DATA"
test -s "$TASK_IMPORTANCE"
echo "$TASK_IMPORTANCE_SHA  $TASK_IMPORTANCE" | sha256sum -c -
test ! -e "$TASK_RSE_OUTPUT"
test ! -e "$TASK_HASHED_OUTPUT"
test ! -e "$TASK_LOG_DIR"

eval "$("$TASK_CONDA" shell.bash hook)"
conda activate sparse_emb
test "$CONDA_DEFAULT_ENV" = sparse_emb
test "$(command -v python3.11)" = "$TASK_PYTHON"
"$TASK_PYTHON" - <<'PY'
import accelerate
import datasets
import torch
import transformers

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
assert all("B200" in torch.cuda.get_device_name(i) for i in range(8))
print(
    f"environment=OK torch={torch.__version__} "
    f"transformers={transformers.__version__} datasets={datasets.__version__} "
    f"accelerate={accelerate.__version__}"
)
PY

export SPARSE_EMB_PYTHON="$TASK_PYTHON"
export SPARSE_EMB_MODEL_DIR="$TASK_MODEL"
export SPARSE_EMB_DATA_DIR="$TASK_DATA"
export SPARSE_EMB_OUTPUT_BASE="$TASK_OUTPUT_BASE"
export PRODUCT_CODE_IMPORTANCE_PATH="$TASK_IMPORTANCE"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0

"$TASK_PYTHON" - "$TASK_RSE_OUTPUT" "$TASK_HASHED_OUTPUT" <<'PY'
import sys

from run_experiments import EXPERIMENT_COMMANDS, missing_input_files

expected = {
    23: ("residual_subspace_experts_tied_g12_r120_q80", sys.argv[1]),
    24: ("product_code_hashed_h2048", sys.argv[2]),
}
for index, (name, output) in expected.items():
    experiment = EXPERIMENT_COMMANDS[index]
    assert experiment["name"] == name, (index, experiment["name"])
    assert experiment["output_dir"] == output, (index, experiment["output_dir"])
    assert experiment["require_fresh_output"] is True
    assert not missing_input_files(experiment), missing_input_files(experiment)
    print(f"selected_experiment={index}:{name} output={output} cmd={experiment['cmd']}")
PY

echo '=== verify current all-GPU workload is exactly the correct burn ==='
echo "$TASK_BURN_SHA  $TASK_BURN" | sha256sum -c -
test -s "$TASK_CURRENT_BURN_LOG"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader
mapfile -t TASK_GPU_PIDS < <(gpu_pids)
test "${#TASK_GPU_PIDS[@]}" -eq 8
TASK_BURN_LAUNCHER=''
for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
  case "$TASK_PID" in
    *[!0-9]*|'') echo "ERROR: invalid GPU PID: $TASK_PID" >&2; exit 1 ;;
    1) echo 'ERROR: refusing to signal PID 1' >&2; exit 1 ;;
  esac
  TASK_PPID="$(awk '/^PPid:/ {print $2}' "/proc/$TASK_PID/status")"
  test "$TASK_PPID" != 1
  if [[ -z "$TASK_BURN_LAUNCHER" ]]; then
    TASK_BURN_LAUNCHER="$TASK_PPID"
  else
    test "$TASK_PPID" = "$TASK_BURN_LAUNCHER"
  fi
done
TASK_BURN_LAUNCHER_CMD="$(tr '\0' ' ' <"/proc/$TASK_BURN_LAUNCHER/cmdline")"
[[ "$TASK_BURN_LAUNCHER_CMD" == *"$TASK_BURN"* ]]
for TASK_MARKER in \
  gpu_burn_ready world_size=8 collective_probe_sum=36 \
  comm_total_mib=1137 comm_bucket_mib=25 approx_step_seconds=0.750; do
  TASK_COUNT="$(grep -oF "$TASK_MARKER" "$TASK_CURRENT_BURN_LOG" | wc -l || true)"
  test "$TASK_COUNT" -eq 8
done
grep -Fq 'gpu_burn_progress' "$TASK_CURRENT_BURN_LOG"
echo "verified_burn_launcher=$TASK_BURN_LAUNCHER workers=${TASK_GPU_PIDS[*]}"

echo '=== stop only the eight verified burn workers ==='
kill -9 "${TASK_GPU_PIDS[@]}" 2>/dev/null || true

set +e
(
  set -euo pipefail

  echo '=== wait 30 seconds, then verify all eight GPUs are free ==='
  sleep 30
  assert_gpus_free
  echo 'ALL_EIGHT_GPUS_FREE_AFTER_BURN_STOP'

  echo '=== copy and verify Accelerate config ==='
  mkdir -p "$(dirname "$TASK_ACCELERATE_DEST")"
  cp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_DEST"
  cmp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_DEST"
  grep -Fx 'distributed_type: MULTI_GPU' "$TASK_ACCELERATE_DEST"
  grep -Fx 'mixed_precision: bf16' "$TASK_ACCELERATE_DEST"
  grep -Fx 'num_processes: 8' "$TASK_ACCELERATE_DEST"

  echo '=== wait 30 seconds, then verify GPUs are still free ==='
  sleep 30
  assert_gpus_free
  echo 'ALL_EIGHT_GPUS_FREE_BEFORE_FINAL_TRAINING'

  mkdir -p "$TASK_LOG_DIR"
  echo '=== train RSE then Hashed sequentially; preserve full schedule, stop at checkpoint 10000 ==='
  set +e
  "$TASK_PYTHON" run_experiments.py \
    --experiments 23 24 \
    --stop-at-step 10000 \
    --log-dir "$TASK_LOG_DIR"
  TASK_TRAIN_RC=$?
  set -e
  echo "final_training_runner_exit_code=$TASK_TRAIN_RC"

  echo '=== wait 30 seconds after runner exit ==='
  sleep 30

  echo '=== verify both checkpoint-10000 saves and training logic ==='
  test -s "$TASK_LOG_DIR/experiments.log"
  test -s "$TASK_LOG_DIR/residual_subspace_experts_tied_g12_r120_q80.log"
  test -s "$TASK_LOG_DIR/product_code_hashed_h2048.log"
  grep -F 'residual_subspace_experts_tied_g12_r120_q80: STOPPED at step 10000' "$TASK_LOG_DIR/experiments.log"
  grep -F 'product_code_hashed_h2048: STOPPED at step 10000' "$TASK_LOG_DIR/experiments.log"
  if grep -E -i 'Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError:|NCCL[^[:cntrl:]]*(unhandled|system error|remote process exited|watchdog|collective operation timeout)' \
      "$TASK_LOG_DIR/residual_subspace_experts_tied_g12_r120_q80.log" \
      "$TASK_LOG_DIR/product_code_hashed_h2048.log"; then
    echo 'ERROR: fatal signature found in final training logs' >&2
    exit 1
  fi

  for TASK_OUTPUT in "$TASK_RSE_OUTPUT" "$TASK_HASHED_OUTPUT"; do
    test -s "$TASK_OUTPUT/train_config.json"
    test ! -e "$TASK_OUTPUT/output_head.pt"
    TASK_CHECKPOINT="$TASK_OUTPUT/checkpoint-10000"
    test ! -e "$TASK_CHECKPOINT/output_head.pt"
    for TASK_REQUIRED in config.json model.safetensors trainer_state.json optimizer.pt scheduler.pt \
        embedding.pt rng_state_0.pth rng_state_1.pth rng_state_2.pth rng_state_3.pth \
        rng_state_4.pth rng_state_5.pth rng_state_6.pth rng_state_7.pth; do
      test -s "$TASK_CHECKPOINT/$TASK_REQUIRED"
    done
  done

  "$TASK_PYTHON" - "$TASK_RSE_OUTPUT" "$TASK_HASHED_OUTPUT" "$TASK_IMPORTANCE" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch

from compositional.loading import _build_arm_from_config
from compositional.product_code import ProductCodeEmbed
from compositional.residual_subspace_experts import ResidualSubspaceExpertsEmbed

rse_output = Path(sys.argv[1])
hashed_output = Path(sys.argv[2])
importance_path = sys.argv[3]

def load_and_validate_common(output):
    checkpoint = output / "checkpoint-10000"
    trainer = json.loads((checkpoint / "trainer_state.json").read_text())
    assert trainer["global_step"] == 10000, trainer["global_step"]
    losses = [row for row in trainer["log_history"] if "loss" in row]
    assert losses, output
    for row in losses:
        for key, value in row.items():
            if key in {"loss", "grad_norm", "learning_rate", "div_loss"}:
                assert math.isfinite(float(value)), (output, key, value)
    saved = json.loads((output / "train_config.json").read_text())
    training = saved["training"]
    assert training["bf16"] is True
    assert training["per_device_train_batch_size"] == 16
    assert training["gradient_accumulation_steps"] == 4
    assert training["num_train_epochs"] == 1
    assert training["max_steps"] == -1
    assert training["seed"] == 42
    assert training["learning_rate"] == 3e-4
    assert training["warmup_steps"] == 500
    assert training["weight_decay"] == 0.1
    assert training["adam_beta1"] == 0.9
    assert training["adam_beta2"] == 0.95
    assert training["save_steps"] == 250
    assert saved["data"]["block_size"] == 2048
    comp = saved["compositional"]
    assert comp["tie_output"] is True
    assert comp["independent_lowrank_output"] is False
    state = torch.load(checkpoint / "embedding.pt", map_location="cpu", weights_only=True)
    for name, tensor in state.items():
        if tensor.is_floating_point():
            assert torch.isfinite(tensor).all(), (output, name)
    return comp, state, losses

rse_config, rse_state, rse_losses = load_and_validate_common(rse_output)
assert rse_config["arm"] == "residual_subspace_experts"
assert rse_config["rse_base_rank"] == 120
assert rse_config["rse_expert_rank"] == 80
assert rse_config["rse_num_experts"] == 12
assert rse_config["rse_router_dim"] == 32
assert rse_config["rse_top_k"] == 2
assert rse_config["rse_router_temperature"] == 1.0
assert rse_config["lambda_div"] == 0.01
assert ResidualSubspaceExpertsEmbed.structure_from_state(rse_state) == {
    "vocab_size": 151936,
    "embed_dim": 1024,
    "base_rank": 120,
    "expert_rank": 80,
    "num_experts": 12,
    "router_dim": 32,
    "top_k": 2,
    "router_temperature": 1.0,
}
rse_module = _build_arm_from_config(rse_config, 151936, 1024, state=rse_state)
rse_module.load_state_dict(rse_state, strict=True)
assert rse_module.parameter_count == 19_471_968
assert rse_state["expert_down_weight"].abs().max().item() > 0.0
assert rse_state["expert_down_bias"].abs().max().item() > 0.0
with torch.no_grad():
    rse_probe = rse_module.materialize(torch.tensor([0, 1, 2048, 50000, 151935]))
assert tuple(rse_probe.shape) == (5, 1024)
assert torch.isfinite(rse_probe).all()

hashed_config, hashed_state, hashed_losses = load_and_validate_common(hashed_output)
assert hashed_config["arm"] == "product_code"
assert hashed_config["product_code_assignment"] == "hashed"
assert hashed_config["product_code_head_size"] == 2048
assert hashed_config["product_code_num_hashes"] == 4
assert hashed_config["product_code_num_buckets"] == 4096
assert hashed_config["product_code_importance_path"] == importance_path
structure = ProductCodeEmbed.structure_from_state(hashed_state)
assert {key: structure[key] for key in (
    "vocab_size", "embed_dim", "head_size", "num_hashes", "num_buckets"
)} == {
    "vocab_size": 151936,
    "embed_dim": 1024,
    "head_size": 2048,
    "num_hashes": 4,
    "num_buckets": 4096,
}
hashed_module = _build_arm_from_config(hashed_config, 151936, 1024, state=hashed_state)
hashed_module.load_state_dict(hashed_state, strict=True)
assert hashed_module.parameter_count == 19_474_944
assert hashed_state["gate_offsets"].abs().max().item() > 0.0
all_ids = torch.cat((hashed_state["head_ids"], hashed_state["tail_ids"]))
assert torch.equal(torch.sort(all_ids).values, torch.arange(151936))
with torch.no_grad():
    probe_ids = torch.cat((hashed_module.head_ids[:3], hashed_module.tail_ids[:3]))
    hashed_probe = hashed_module.materialize(probe_ids)
assert tuple(hashed_probe.shape) == (6, 1024)
assert torch.isfinite(hashed_probe).all()

print(f"RSE_CHECKPOINT_10000_PASS last_loss={rse_losses[-1]}")
print(f"HASHED_CHECKPOINT_10000_PASS last_loss={hashed_losses[-1]}")
print("FINAL_RSE_HASHED_TRAINING_ARTIFACT_AUDIT_PASS")
PY

  test "$TASK_TRAIN_RC" -eq 0
  assert_gpus_free
  echo 'ALL_EIGHT_GPUS_FREE_AFTER_FINAL_TRAINING_AND_AUDIT'
)
TASK_WORK_RC=$?
set -e
echo "final_training_and_audit_exit_code=$TASK_WORK_RC"

echo '=== ensure no workload remains before restoring burn ==='
for TASK_WAIT_INDEX in $(seq 1 12); do
  mapfile -t TASK_AFTER_TRAIN_PIDS < <(gpu_pids)
  (( ${#TASK_AFTER_TRAIN_PIDS[@]} == 0 )) && break
  echo "waiting_for_gpu_pids=${TASK_AFTER_TRAIN_PIDS[*]}"
  sleep 5
done
if (( ${#TASK_AFTER_TRAIN_PIDS[@]} )); then
  echo "ERROR: refusing to start burn while GPU PIDs remain: ${TASK_AFTER_TRAIN_PIDS[*]}" >&2
  for TASK_PID in "${TASK_AFTER_TRAIN_PIDS[@]}"; do
    printf 'remaining_gpu_pid=%s cmd=' "$TASK_PID" >&2
    tr '\0' ' ' <"/proc/$TASK_PID/cmdline" >&2 2>/dev/null || true
    echo >&2
  done
  nvidia-smi
  exit 1
fi
assert_gpus_free

echo '=== restart and verify correct communicating burn on all eight GPUs ==='
test -x /usr/bin/python3
echo "$TASK_BURN_SHA  $TASK_BURN" | sha256sum -c -
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  nohup /usr/bin/python3 "$TASK_BURN" >"$TASK_NEW_BURN_LOG" 2>&1 &
TASK_NEW_BURN_LAUNCHER=$!
echo "$TASK_NEW_BURN_LAUNCHER" >"$TASK_BURN_PID_FILE"
sleep 30
mapfile -t TASK_NEW_BURN_PIDS < <(gpu_pids)
test "${#TASK_NEW_BURN_PIDS[@]}" -eq 8
for TASK_PID in "${TASK_NEW_BURN_PIDS[@]}"; do
  TASK_PPID="$(awk '/^PPid:/ {print $2}' "/proc/$TASK_PID/status")"
  test "$TASK_PPID" = "$TASK_NEW_BURN_LAUNCHER"
done
for TASK_MARKER in \
  gpu_burn_ready world_size=8 collective_probe_sum=36 \
  comm_total_mib=1137 comm_bucket_mib=25 approx_step_seconds=0.750; do
  TASK_COUNT="$(grep -oF "$TASK_MARKER" "$TASK_NEW_BURN_LOG" | wc -l || true)"
  test "$TASK_COUNT" -eq 8
done
grep -Fq 'gpu_burn_progress' "$TASK_NEW_BURN_LOG"
TASK_GPU_STATE="$(
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits
)"
printf '%s\n' "$TASK_GPU_STATE"
"$TASK_PYTHON" - "$TASK_GPU_STATE" <<'PY'
import sys

rows = []
for line in sys.argv[1].splitlines():
    fields = [field.strip() for field in line.split(",")]
    rows.append((int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])))
assert len(rows) == 8, rows
assert {row[0] for row in rows} == set(range(8))
for gpu, used, total, utilization in rows:
    assert used >= 140_000, (gpu, used, total, utilization)
    assert used < total, (gpu, used, total, utilization)
print("ALL_EIGHT_RESTORED_BURNS_USE_MOST_GPU_MEMORY")
PY
echo "new_burn_launcher=$TASK_NEW_BURN_LAUNCHER workers=${TASK_NEW_BURN_PIDS[*]}"
echo 'CORRECT_COMMUNICATING_BURN_RUNNING_ON_ALL_EIGHT_GPUS'

test "$TASK_WORK_RC" -eq 0
echo 'TH2 FINAL RSE AND HASHED 10K TRAINING COMPLETE'
