#1 +240+a
#th2-rse-production-shape-smoke-20260831-a01
set -euo pipefail

TASK_PROJECT=/mnt/local/@PROJECT@
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_MODEL=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_DATA=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/train
TASK_OUTPUT=/mnt/local/_outputs/@PROJECT@/residual_subspace_experts_prodshape_smoke_20260831_a01
TASK_BURN=/tmp/llm_pretrain_burn.py
TASK_BURN_SHA=2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174
TASK_CURRENT_BURN_LOG=/tmp/llm_pretrain_burn_after_product_code_smoke.log
TASK_NEW_BURN_LOG=/tmp/llm_pretrain_burn_after_rse_smoke.log
TASK_BURN_PID_FILE=/tmp/llm_pretrain_burn_launcher.pid

cd "$TASK_PROJECT"

echo '=== source, input, output, and environment preflight before touching GPUs ==='
date -u
hostname
pwd
echo 'ffc30429ebaaed3561f97c4706b8d6deb674e4097a16457c994e1692981db326  compositional/residual_subspace_experts.py' | sha256sum -c -
echo '58a4dfdbe46d45ce674a1f6d3cd8501941b0e8eccda2dc1a047eaeda3fa4bf35  train_compositional.py' | sha256sum -c -
echo '7be9957197d39b454cfcdda08989588e545b6487c9b2302362e96afcdcd581b9  scripts/train_residual_subspace_experts_tied.sh' | sha256sum -c -
echo '923db7f20a2df3d051180f67f9bea1f30c84c804651e313fa9961a9fd17a57e5  resources/accelerate_config.yaml' | sha256sum -c -
test -d "$TASK_MODEL"
test -s "$TASK_MODEL/config.json"
test -d "$TASK_DATA"
test ! -e "$TASK_OUTPUT"
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate sparse_emb
test "$(command -v python3.11)" = "$TASK_PYTHON"
mkdir -p "$HOME/.cache/huggingface/accelerate"
cp resources/accelerate_config.yaml "$HOME/.cache/huggingface/accelerate/default_config.yaml"
cmp -s resources/accelerate_config.yaml "$HOME/.cache/huggingface/accelerate/default_config.yaml"
echo 'RSE production-shape smoke: Qwen3-0.6B, seq=2048, batch=16/GPU, accum=4, 8xB200, 3 optimizer steps'

echo '=== verify current all-GPU workload is exactly the restored runner burn ==='
echo "$TASK_BURN_SHA  $TASK_BURN" | sha256sum -c -
test -s "$TASK_CURRENT_BURN_LOG"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader
mapfile -t TASK_GPU_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d;s/[[:space:]]//g' | sort -nu
)
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
TASK_BURN_LAUNCHER_CMD="$(tr '\0' ' ' < "/proc/$TASK_BURN_LAUNCHER/cmdline")"
[[ "$TASK_BURN_LAUNCHER_CMD" == *"$TASK_BURN"* ]]
for TASK_MARKER in \
  gpu_burn_ready world_size=8 collective_probe_sum=36 \
  comm_total_mib=1137 comm_bucket_mib=25 approx_step_seconds=0.750; do
  TASK_COUNT="$(grep -oF "$TASK_MARKER" "$TASK_CURRENT_BURN_LOG" | wc -l || true)"
  test "$TASK_COUNT" -eq 8
done
echo "verified_burn_launcher=$TASK_BURN_LAUNCHER workers=${TASK_GPU_PIDS[*]}"

echo '=== stop only verified GPU compute workers ==='
kill -9 "${TASK_GPU_PIDS[@]}" 2>/dev/null || true
sleep 30
mapfile -t TASK_REMAINING_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d;s/[[:space:]]//g' | sort -nu
)
if (("${#TASK_REMAINING_PIDS[@]}")); then
  echo "ERROR: GPU compute PIDs remain after stopping burn: ${TASK_REMAINING_PIDS[*]}" >&2
  nvidia-smi
  exit 1
fi
nvidia-smi
echo 'ALL_EIGHT_GPUS_FREE_AFTER_BURN_STOP'

mkdir -p "$TASK_OUTPUT"
echo '=== run full production-shape RSE smoke ==='
set +e
(
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export WANDB_PROJECT=sparse_embedding
  export WANDB_MODE=offline
  export NCCL_NVLS_ENABLE=0
  export TOKENIZERS_PARALLELISM=false
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  "$TASK_PYTHON" -m accelerate.commands.launch train_compositional.py \
    --config_name "$TASK_MODEL" \
    --tokenizer_name "$TASK_MODEL" \
    --data_dir "$TASK_DATA" \
    --block_size 2048 \
    --preprocessing_num_workers 160 \
    --seed 42 \
    --bf16 \
    --ddp_timeout 21600 \
    --ddp_find_unused_parameters false \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_steps 3 \
    --learning_rate 3e-4 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
    --warmup_steps 500 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --save_steps 1 \
    --save_total_limit 3 \
    --dataloader_num_workers 8 \
    --report_to wandb \
    --output_dir "$TASK_OUTPUT" \
    --run_name rse-prodshape-smoke-20260831-a01 \
    --arm residual_subspace_experts \
    --rse_base_rank 120 \
    --rse_expert_rank 80 \
    --rse_num_experts 12 \
    --rse_router_dim 32 \
    --rse_top_k 2 \
    --rse_router_temperature 1.0 \
    --lambda_div 0.01 \
    --tie_output \
    >"$TASK_OUTPUT/train.log" 2>&1 &
  TASK_TRAIN_PID=$!

  (
    while kill -0 "$TASK_TRAIN_PID" 2>/dev/null; do
      date -u '+%Y-%m-%dT%H:%M:%SZ'
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv,noheader,nounits
      sleep 5
    done
  ) >"$TASK_OUTPUT/gpu_monitor.log" 2>&1 &
  TASK_MONITOR_PID=$!

  set +e
  wait "$TASK_TRAIN_PID"
  TASK_TRAIN_RC=$?
  set -e
  wait "$TASK_MONITOR_PID" 2>/dev/null || true
  echo "accelerate_exit_code=$TASK_TRAIN_RC"
  tail -140 "$TASK_OUTPUT/train.log"
  test "$TASK_TRAIN_RC" -eq 0

  if grep -E -i 'Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError:|NCCL[^[:cntrl:]]*(unhandled|system error|remote process exited|watchdog|collective operation timeout)' "$TASK_OUTPUT/train.log"; then
    echo 'ERROR: fatal signature found in RSE smoke log' >&2
    exit 1
  fi

  TASK_CHECKPOINT="$TASK_OUTPUT/checkpoint-3"
  for TASK_REQUIRED in config.json model.safetensors trainer_state.json optimizer.pt scheduler.pt \
      embedding.pt rng_state_0.pth rng_state_1.pth rng_state_2.pth rng_state_3.pth \
      rng_state_4.pth rng_state_5.pth rng_state_6.pth rng_state_7.pth; do
    test -s "$TASK_CHECKPOINT/$TASK_REQUIRED"
  done
  test -s "$TASK_OUTPUT/train_config.json"
  test -s "$TASK_OUTPUT/train_results.json"
  test -s "$TASK_OUTPUT/embedding.pt"
  test ! -e "$TASK_OUTPUT/output_head.pt"
  test ! -e "$TASK_CHECKPOINT/output_head.pt"

  "$TASK_PYTHON" - "$TASK_CHECKPOINT" "$TASK_OUTPUT/train_config.json" \
    "$TASK_OUTPUT/train_results.json" "$TASK_OUTPUT/gpu_monitor.log" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

import torch

from compositional.loading import _build_arm_from_config
from compositional.residual_subspace_experts import (
    ResidualSubspaceExpertsEmbed,
)
from compositional.tied_head import make_tied_head

checkpoint = Path(sys.argv[1])
config_path = Path(sys.argv[2])
results_path = Path(sys.argv[3])
monitor_path = Path(sys.argv[4])

trainer = json.loads((checkpoint / "trainer_state.json").read_text())
assert trainer["global_step"] == 3, trainer["global_step"]
loss_rows = [row for row in trainer["log_history"] if "loss" in row]
assert len(loss_rows) >= 3, loss_rows
for row in loss_rows:
    for key in ("loss", "grad_norm", "learning_rate", "div_loss"):
        assert math.isfinite(float(row[key])), (key, row[key])
assert any(float(row["div_loss"]) > 0.0 for row in loss_rows)
results = json.loads(results_path.read_text())
for key, value in results.items():
    if isinstance(value, (int, float)):
        assert math.isfinite(float(value)), (key, value)

config = json.loads(config_path.read_text())["compositional"]
assert config["arm"] == "residual_subspace_experts"
assert config["tie_output"] is True
assert config["independent_lowrank_output"] is False
assert config["rse_base_rank"] == 120
assert config["rse_expert_rank"] == 80
assert config["rse_num_experts"] == 12
assert config["rse_router_dim"] == 32
assert config["rse_top_k"] == 2
assert config["rse_router_temperature"] == 1.0
assert config["lambda_div"] == 0.01

state = torch.load(checkpoint / "embedding.pt", map_location="cpu", weights_only=True)
structure = ResidualSubspaceExpertsEmbed.structure_from_state(state)
assert structure == {
    "vocab_size": 151936,
    "embed_dim": 1024,
    "base_rank": 120,
    "expert_rank": 80,
    "num_experts": 12,
    "router_dim": 32,
    "top_k": 2,
    "router_temperature": 1.0,
}
expected_shapes = {
    "token_factors": (151936, 120),
    "base_proj.weight": (1024, 120),
    "base_proj.bias": (1024,),
    "expert_down_weight": (12, 80, 120),
    "expert_down_bias": (12, 80),
    "expert_up_weight": (12, 1024, 80),
    "expert_up_bias": (12, 1024),
    "router_proj.weight": (32, 120),
    "router_proj.bias": (32,),
    "expert_keys": (12, 32),
}
for name, shape in expected_shapes.items():
    assert tuple(state[name].shape) == shape, (name, state[name].shape)
    assert torch.isfinite(state[name]).all(), name
assert state["expert_down_weight"].abs().max().item() > 0.0
assert state["expert_down_bias"].abs().max().item() > 0.0
assert state["expert_up_bias"].abs().max().item() > 0.0

module = _build_arm_from_config(config, 151936, 1024, state=state)
module.load_state_dict(state, strict=True)
module.eval()
assert module.parameter_count == 19_471_968
device = torch.device("cuda:0")
module.to(device=device, dtype=torch.float32)
with torch.no_grad():
    probe_ids = torch.tensor([0, 1, 2, 1024, 50000, 151935], device=device)
    probe = module.materialize(probe_ids)
    assert tuple(probe.shape) == (6, 1024)
    assert torch.isfinite(probe).all()
    hidden = torch.randn(1, 2, 1024, device=device)
    tied_head = make_tied_head(module, "residual_subspace_experts", 151936)
    logits = tied_head(hidden)
    reference = hidden @ module.materialize().T
    torch.testing.assert_close(logits, reference, rtol=2e-4, atol=2e-4)
    assert tuple(logits.shape) == (1, 2, 151936)
del logits, reference, tied_head, module
torch.cuda.synchronize(device)
torch.cuda.empty_cache()

rows = []
for line in monitor_path.read_text().splitlines():
    if re.match(r"^\d+,", line):
        parts = [part.strip() for part in line.split(",")]
        rows.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
assert rows, "no GPU monitor samples"
seen = {row[0] for row in rows}
assert seen == set(range(8)), seen
peak_by_gpu = {
    gpu: max(row[1] for row in rows if row[0] == gpu)
    for gpu in sorted(seen)
}
assert all(value > 0 for value in peak_by_gpu.values())
print("trainer_global_step=3")
print(f"finite_loss_rows={len(loss_rows)} last_loss={loss_rows[-1]}")
print(f"train_results={results}")
print(f"expert_down_abs_max={state['expert_down_weight'].abs().max().item():.9g}")
print(f"expert_up_bias_abs_max={state['expert_up_bias'].abs().max().item():.9g}")
print("peak_memory_mib=" + json.dumps(peak_by_gpu, sort_keys=True))
print("RSE_PRODUCTION_SHAPE_CHECKPOINT_AND_EXACT_TIED_HEAD_PASS")
PY
)
TASK_SMOKE_RC=$?
set -e
echo "rse_smoke_exit_code=$TASK_SMOKE_RC"

echo '=== ensure all RSE and validation GPU processes have exited ==='
for TASK_WAIT_INDEX in $(seq 1 12); do
  mapfile -t TASK_AFTER_TEST_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      | sed '/^[[:space:]]*$/d;s/[[:space:]]//g' | sort -nu
  )
  (("${#TASK_AFTER_TEST_PIDS[@]}" == 0)) && break
  echo "waiting_for_test_gpu_pids=${TASK_AFTER_TEST_PIDS[*]}"
  sleep 5
done
if (("${#TASK_AFTER_TEST_PIDS[@]}")); then
  echo "ERROR: GPU compute PIDs remain after RSE test: ${TASK_AFTER_TEST_PIDS[*]}" >&2
  TASK_ALL_OWNED=1
  for TASK_PID in "${TASK_AFTER_TEST_PIDS[@]}"; do
    TASK_CMD="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline" 2>/dev/null || true)"
    echo "remaining_gpu_pid=$TASK_PID cmd=$TASK_CMD"
    if [[ "$TASK_CMD" != *train_compositional.py* ]] || \
       [[ "$TASK_CMD" != *"--arm residual_subspace_experts"* ]]; then
      TASK_ALL_OWNED=0
    fi
  done
  if [[ "$TASK_ALL_OWNED" -eq 1 ]]; then
    kill -9 "${TASK_AFTER_TEST_PIDS[@]}" 2>/dev/null || true
    sleep 5
  else
    echo 'ERROR: refusing to kill an unidentified GPU process' >&2
    exit 1
  fi
fi
mapfile -t TASK_FINAL_TEST_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d;s/[[:space:]]//g' | sort -nu
)
test "${#TASK_FINAL_TEST_PIDS[@]}" -eq 0
nvidia-smi
echo 'ALL_EIGHT_GPUS_FREE_AFTER_RSE_SMOKE'

echo '=== restart and verify correct communicating burn on all eight GPUs ==='
test -x /usr/bin/python3
echo "$TASK_BURN_SHA  $TASK_BURN" | sha256sum -c -
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  nohup /usr/bin/python3 "$TASK_BURN" >"$TASK_NEW_BURN_LOG" 2>&1 &
TASK_NEW_BURN_LAUNCHER=$!
echo "$TASK_NEW_BURN_LAUNCHER" >"$TASK_BURN_PID_FILE"
sleep 30
mapfile -t TASK_NEW_BURN_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d;s/[[:space:]]//g' | sort -nu
)
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
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader
echo "new_burn_launcher=$TASK_NEW_BURN_LAUNCHER workers=${TASK_NEW_BURN_PIDS[*]}"
echo 'CORRECT_ALL_EIGHT_GPU_BURN_RESTARTED_AFTER_RSE_SMOKE'

test "$TASK_SMOKE_RC" -eq 0
echo 'TH2 RSE PRODUCTION-SHAPE TEST COMPLETE'
