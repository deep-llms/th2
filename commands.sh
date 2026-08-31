#1 +240+a
#th2-train-final-rse-b8a8-then-hashed-10k-20260831-a02
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
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/final_rse_b8a8_hashed_10k_20260831_a02"
TASK_ACCELERATE_SOURCE="$TASK_PROJECT/resources/accelerate_config.yaml"
TASK_ACCELERATE_DEST=/mnt/local/.cache/huggingface/accelerate/default_config.yaml
TASK_BURN=/tmp/llm_pretrain_burn.py
TASK_BURN_SHA=2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174
TASK_BURN_LOG=/tmp/llm_pretrain_burn_after_final_rse_b8a8_hashed_10k_a02.log
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
echo '=== immutable source, input, and output preflight ==='
date -u
hostname
pwd
echo '26dfd72c83b3c23aad9d21e13f51b7ce1f8914b9a3a4c325de5bf371cda91529  scripts/train_residual_subspace_experts_tied.sh' | sha256sum -c -
echo 'f4d4963c0fadc578933e429fdf436812aa8f208826d66da9ee51e95a89cace2d  scripts/train_product_code_tied.sh' | sha256sum -c -
echo '726a15849995136c313a2e4c888988385216387b4ac3f1f5a57e61f6ea86e171  run_experiments.py' | sha256sum -c -
echo '58a4dfdbe46d45ce674a1f6d3cd8501941b0e8eccda2dc1a047eaeda3fa4bf35  train_compositional.py' | sha256sum -c -
echo 'f2f5ea64b63d933d7e0609bd27f3226bf4c8abcf7bcf582a5907557ecb2ab606  compositional/residual_subspace_experts.py' | sha256sum -c -
echo 'b17de9f5ab2211c0825922bfaf8f23ed9ecf98c7cae29cc3973074c5dac843a1  compositional/product_code.py' | sha256sum -c -
echo '923db7f20a2df3d051180f67f9bea1f30c84c804651e313fa9961a9fd17a57e5  resources/accelerate_config.yaml' | sha256sum -c -
echo "$TASK_BURN_SHA  $TASK_BURN" | sha256sum -c -
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

export SPARSE_EMB_PYTHON="$TASK_PYTHON"
export SPARSE_EMB_MODEL_DIR="$TASK_MODEL"
export SPARSE_EMB_DATA_DIR="$TASK_DATA"
export SPARSE_EMB_OUTPUT_BASE="$TASK_OUTPUT_BASE"
export PRODUCT_CODE_IMPORTANCE_PATH="$TASK_IMPORTANCE"
export RSE_PER_DEVICE_TRAIN_BATCH_SIZE=8
export RSE_GRADIENT_ACCUMULATION_STEPS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
    assert experiment["name"] == name
    assert experiment["output_dir"] == output
    assert experiment["require_fresh_output"] is True
    assert not missing_input_files(experiment), missing_input_files(experiment)
    print(f"selected={index}:{name} output={output} cmd={experiment['cmd']}")
print("RSE schedule: 8 GPUs x batch 8 x accumulation 8 = global batch 512")
print("Hashed schedule: 8 GPUs x batch 16 x accumulation 4 = global batch 512")
PY

echo '=== check all eight GPUs are free ==='
assert_gpus_free
echo 'ALL_EIGHT_GPUS_FREE_INITIAL_CHECK'
sleep 30
assert_gpus_free
echo 'ALL_EIGHT_GPUS_FREE_AFTER_30_SECONDS'

echo '=== copy and verify Accelerate configuration ==='
mkdir -p "$(dirname "$TASK_ACCELERATE_DEST")"
cp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_DEST"
cmp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_DEST"
grep -Fx 'distributed_type: MULTI_GPU' "$TASK_ACCELERATE_DEST"
grep -Fx 'mixed_precision: bf16' "$TASK_ACCELERATE_DEST"
grep -Fx 'num_processes: 8' "$TASK_ACCELERATE_DEST"
sleep 30
assert_gpus_free
echo 'ALL_EIGHT_GPUS_FREE_IMMEDIATELY_BEFORE_TRAINING'

set +e
(
  set -euo pipefail
  mkdir -p "$TASK_LOG_DIR"
  echo '=== train memory-safe RSE then Hashed sequentially to checkpoint 10000 ==='
  set +e
  "$TASK_PYTHON" run_experiments.py \
    --experiments 23 24 \
    --stop-at-step 10000 \
    --log-dir "$TASK_LOG_DIR"
  TASK_TRAIN_RC=$?
  set -e
  echo "training_runner_exit_code=$TASK_TRAIN_RC"
  sleep 30

  test -s "$TASK_LOG_DIR/experiments.log"
  test -s "$TASK_LOG_DIR/residual_subspace_experts_tied_g12_r120_q80.log"
  test -s "$TASK_LOG_DIR/product_code_hashed_h2048.log"
  grep -F 'residual_subspace_experts_tied_g12_r120_q80: STOPPED at step 10000' \
    "$TASK_LOG_DIR/experiments.log"
  grep -F 'product_code_hashed_h2048: STOPPED at step 10000' \
    "$TASK_LOG_DIR/experiments.log"
  if grep -E -i 'Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError:|NCCL[^[:cntrl:]]*(unhandled|system error|remote process exited|watchdog|collective operation timeout)' \
      "$TASK_LOG_DIR/residual_subspace_experts_tied_g12_r120_q80.log" \
      "$TASK_LOG_DIR/product_code_hashed_h2048.log"; then
    echo 'ERROR: fatal signature found in final training logs' >&2
    exit 1
  fi

  for TASK_OUTPUT in "$TASK_RSE_OUTPUT" "$TASK_HASHED_OUTPUT"; do
    TASK_CHECKPOINT="$TASK_OUTPUT/checkpoint-10000"
    test -s "$TASK_OUTPUT/train_config.json"
    test ! -e "$TASK_OUTPUT/output_head.pt"
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

rse_output, hashed_output = map(Path, sys.argv[1:3])
importance_path = sys.argv[3]

def common(output, expected_batch, expected_accum):
    checkpoint = output / "checkpoint-10000"
    trainer = json.loads((checkpoint / "trainer_state.json").read_text())
    assert trainer["global_step"] == 10000
    losses = [row for row in trainer["log_history"] if "loss" in row]
    assert losses
    for row in losses:
        for key in ("loss", "grad_norm", "learning_rate"):
            assert math.isfinite(float(row[key])), (output, key, row[key])
        if "div_loss" in row:
            assert math.isfinite(float(row["div_loss"]))
    saved = json.loads((output / "train_config.json").read_text())
    training = saved["training"]
    assert training["bf16"] is True
    assert training["per_device_train_batch_size"] == expected_batch
    assert training["gradient_accumulation_steps"] == expected_accum
    assert 8 * expected_batch * expected_accum == 512
    assert training["num_train_epochs"] == 1
    assert training["max_steps"] == -1
    assert training["seed"] == 42
    assert training["learning_rate"] == 3e-4
    assert training["warmup_steps"] == 500
    assert training["weight_decay"] == 0.1
    assert training["save_steps"] == 250
    assert saved["data"]["block_size"] == 2048
    config = saved["compositional"]
    assert config["tie_output"] is True
    assert config["independent_lowrank_output"] is False
    state = torch.load(checkpoint / "embedding.pt", map_location="cpu", weights_only=True)
    for name, tensor in state.items():
        if tensor.is_floating_point():
            assert torch.isfinite(tensor).all(), (output, name)
    return config, state, losses

rse_config, rse_state, rse_losses = common(rse_output, 8, 8)
assert rse_config["arm"] == "residual_subspace_experts"
assert rse_config["rse_base_rank"] == 120
assert rse_config["rse_expert_rank"] == 80
assert rse_config["rse_num_experts"] == 12
assert rse_config["rse_router_dim"] == 32
assert rse_config["rse_top_k"] == 2
assert rse_config["lambda_div"] == 0.01
rse_structure = ResidualSubspaceExpertsEmbed.structure_from_state(rse_state)
assert rse_structure["vocab_size"] == 151936
assert rse_structure["embed_dim"] == 1024
rse_module = _build_arm_from_config(rse_config, 151936, 1024, state=rse_state)
rse_module.load_state_dict(rse_state, strict=True)
assert rse_module.parameter_count == 19_471_968
assert rse_state["expert_down_weight"].abs().max().item() > 0.0

hashed_config, hashed_state, hashed_losses = common(hashed_output, 16, 4)
assert hashed_config["arm"] == "product_code"
assert hashed_config["product_code_assignment"] == "hashed"
assert hashed_config["product_code_head_size"] == 2048
assert hashed_config["product_code_num_hashes"] == 4
assert hashed_config["product_code_num_buckets"] == 4096
assert hashed_config["product_code_importance_path"] == importance_path
hashed_structure = ProductCodeEmbed.structure_from_state(hashed_state)
assert hashed_structure["vocab_size"] == 151936
assert hashed_structure["embed_dim"] == 1024
hashed_module = _build_arm_from_config(hashed_config, 151936, 1024, state=hashed_state)
hashed_module.load_state_dict(hashed_state, strict=True)
assert hashed_module.parameter_count == 19_474_944
assert hashed_state["gate_offsets"].abs().max().item() > 0.0

print(f"RSE_B8A8_CHECKPOINT_10000_PASS last_loss={rse_losses[-1]}")
print(f"HASHED_CHECKPOINT_10000_PASS last_loss={hashed_losses[-1]}")
print("BOTH_FINAL_EXPERIMENTS_VALIDATED")
PY
  test "$TASK_TRAIN_RC" -eq 0
  assert_gpus_free
  echo 'ALL_EIGHT_GPUS_FREE_AFTER_BOTH_FINAL_EXPERIMENTS'
)
TASK_WORK_RC=$?
set -e
echo "training_and_audit_exit_code=$TASK_WORK_RC"

echo '=== wait for all training workers to release GPUs ==='
for TASK_WAIT_INDEX in $(seq 1 12); do
  mapfile -t TASK_AFTER_TRAIN_PIDS < <(gpu_pids)
  (( ${#TASK_AFTER_TRAIN_PIDS[@]} == 0 )) && break
  echo "waiting_for_gpu_pids=${TASK_AFTER_TRAIN_PIDS[*]}"
  sleep 5
done
if (( ${#TASK_AFTER_TRAIN_PIDS[@]} )); then
  echo "ERROR: refusing to start burn while GPU PIDs remain: ${TASK_AFTER_TRAIN_PIDS[*]}" >&2
  nvidia-smi
  exit 1
fi
assert_gpus_free

echo '=== start and verify correct communicating high-memory burn on all eight GPUs ==='
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  nohup /usr/bin/python3 "$TASK_BURN" >"$TASK_BURN_LOG" 2>&1 &
TASK_BURN_LAUNCHER=$!
echo "$TASK_BURN_LAUNCHER" >"$TASK_BURN_PID_FILE"
sleep 30
mapfile -t TASK_BURN_PIDS < <(gpu_pids)
test "${#TASK_BURN_PIDS[@]}" -eq 8
for TASK_PID in "${TASK_BURN_PIDS[@]}"; do
  TASK_PPID="$(awk '/^PPid:/ {print $2}' "/proc/$TASK_PID/status")"
  test "$TASK_PPID" = "$TASK_BURN_LAUNCHER"
done
for TASK_MARKER in \
    gpu_burn_ready world_size=8 collective_probe_sum=36 \
    comm_total_mib=1137 comm_bucket_mib=25 approx_step_seconds=0.750; do
  TASK_COUNT="$(grep -oF "$TASK_MARKER" "$TASK_BURN_LOG" | wc -l || true)"
  test "$TASK_COUNT" -eq 8
done
grep -Fq 'gpu_burn_progress' "$TASK_BURN_LOG"
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
print("ALL_EIGHT_BURNS_USE_MOST_GPU_MEMORY")
PY
echo "burn_launcher=$TASK_BURN_LAUNCHER workers=${TASK_BURN_PIDS[*]}"
echo 'CORRECT_COMMUNICATING_BURN_RUNNING_ON_ALL_EIGHT_GPUS'
test "$TASK_WORK_RC" -eq 0
echo 'TH2 FINAL RSE B8A8 AND HASHED 10K TRAINING COMPLETE'
