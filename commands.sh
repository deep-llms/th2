#0
#th2-idle-while-product-code-production-shape-smoke-a02-runs-20260831
set -euo pipefail

TASK_PROJECT=/mnt/local/@PROJECT@
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_MODEL=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_DATA=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/train
TASK_OUTPUT=/mnt/local/_outputs/@PROJECT@/product_code_prodshape_smoke_20260831_a01
TASK_ARTIFACT_DIR=/mnt/local/_outputs/@PROJECT@/product_code_smoke_artifacts
TASK_IMPORTANCE="$TASK_ARTIFACT_DIR/token_importance_langbalanced.npz"
TASK_BURN=/tmp/llm_pretrain_burn.py
TASK_BURN_SHA=2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174
TASK_BURN_LOG=/tmp/llm_pretrain_burn_after_product_code_smoke.log
TASK_BURN_PID_FILE=/tmp/llm_pretrain_burn_launcher.pid

cd "$TASK_PROJECT"

echo '=== identity and intended workload ==='
date -u
hostname
git rev-parse HEAD
echo 'Product Code production-shape smoke: Qwen3-0.6B, seq=2048, batch=16/GPU, accum=4, 8xB200, 2 optimizer steps'

echo '=== verify the current all-GPU workload is exactly the runner burn ==='
echo "$TASK_BURN_SHA  $TASK_BURN" | sha256sum -c -
test -s /tmp/llm_pretrain_burn_all_gpus.log
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
  TASK_COUNT="$(grep -oF "$TASK_MARKER" /tmp/llm_pretrain_burn_all_gpus.log | wc -l || true)"
  test "$TASK_COUNT" -eq 8
done
echo "verified_burn_launcher=$TASK_BURN_LAUNCHER workers=${TASK_GPU_PIDS[*]}"

echo '=== stop only GPU compute workers; never match the burn name ==='
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

echo '=== environment, offline inputs, and accelerate configuration ==='
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate sparse_emb
test "$(command -v python3.11)" = "$TASK_PYTHON"
test -d "$TASK_MODEL"
test -s "$TASK_MODEL/config.json"
test -d "$TASK_DATA"
test -s resources/token_freq_sample10.npz
test -s resources/accelerate_config.yaml
mkdir -p "$HOME/.cache/huggingface/accelerate" "$TASK_ARTIFACT_DIR"
cp resources/accelerate_config.yaml "$HOME/.cache/huggingface/accelerate/default_config.yaml"
cmp -s resources/accelerate_config.yaml "$HOME/.cache/huggingface/accelerate/default_config.yaml"
"$TASK_PYTHON" scripts/make_token_importance.py \
  --source resources/token_freq_sample10.npz \
  --output "$TASK_IMPORTANCE" \
  --head-size 2048
echo '39b15eab8cf213d563dcf5137bb982e836bb8e3beba8e7def8dcddf21fe43594  '"$TASK_IMPORTANCE" | sha256sum -c -
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_OUTPUT"

echo '=== run full production-shape Product Code smoke ==='
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
    --max_steps 2 \
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
    --save_total_limit 2 \
    --dataloader_num_workers 8 \
    --report_to wandb \
    --output_dir "$TASK_OUTPUT" \
    --run_name product-code-prodshape-smoke-20260831-a01 \
    --arm product_code \
    --product_code_head_size 2048 \
    --product_code_num_hashes 4 \
    --product_code_num_buckets 4096 \
    --product_code_assignment hashed \
    --product_code_importance_path "$TASK_IMPORTANCE" \
    --product_code_importance_key counts \
    --product_code_seed 0 \
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
  tail -120 "$TASK_OUTPUT/train.log"
  test "$TASK_TRAIN_RC" -eq 0

  if grep -E -i 'Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError:|NCCL[^[:cntrl:]]*(unhandled|system error|remote process exited|watchdog|collective operation timeout)' "$TASK_OUTPUT/train.log"; then
    echo 'ERROR: fatal signature found in Product Code smoke log' >&2
    exit 1
  fi

  TASK_CHECKPOINT="$TASK_OUTPUT/checkpoint-2"
  for TASK_REQUIRED in config.json model.safetensors trainer_state.json optimizer.pt scheduler.pt \
      embedding.pt rng_state_0.pth rng_state_1.pth rng_state_2.pth rng_state_3.pth \
      rng_state_4.pth rng_state_5.pth rng_state_6.pth rng_state_7.pth; do
    test -s "$TASK_CHECKPOINT/$TASK_REQUIRED"
  done
  test -s "$TASK_OUTPUT/train_config.json"
  test -s "$TASK_OUTPUT/train_results.json"
  test -s "$TASK_OUTPUT/embedding.pt"

  "$TASK_PYTHON" - "$TASK_CHECKPOINT" "$TASK_OUTPUT/train_config.json" "$TASK_OUTPUT/gpu_monitor.log" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

import torch

checkpoint = Path(sys.argv[1])
config_path = Path(sys.argv[2])
monitor_path = Path(sys.argv[3])
trainer = json.loads((checkpoint / "trainer_state.json").read_text())
assert trainer["global_step"] == 2, trainer["global_step"]
loss_rows = [row for row in trainer["log_history"] if "loss" in row]
assert len(loss_rows) >= 2, loss_rows
for row in loss_rows:
    for key in ("loss", "grad_norm", "learning_rate"):
        assert math.isfinite(float(row[key])), (key, row[key])

config = json.loads(config_path.read_text())["compositional"]
assert config["arm"] == "product_code"
assert config["tie_output"] is True
assert config["product_code_assignment"] == "hashed"
assert config["product_code_head_size"] == 2048
assert config["product_code_num_hashes"] == 4
assert config["product_code_num_buckets"] == 4096

state = torch.load(checkpoint / "embedding.pt", map_location="cpu", weights_only=True)
assert tuple(state["E_h"].shape) == (2048, 1024)
for index in range(4):
    assert tuple(state[f"C.{index}"].shape) == (4096, 1024)
assert tuple(state["codes"].shape) == (149888, 4)
assert tuple(state["gate_offsets"].shape) == (149888, 4)
assert tuple(state["head_ids"].shape) == (2048,)
assert tuple(state["tail_ids"].shape) == (149888,)
assert tuple(state["bias"].shape) == (151936,)
assert torch.isfinite(state["E_h"]).all()
assert all(torch.isfinite(state[f"C.{index}"]).all() for index in range(4))
assert torch.isfinite(state["gate_offsets"]).all()
assert state["gate_offsets"].abs().max().item() > 0.0
assert state["codes"].min().item() >= 0
assert state["codes"].max().item() < 4096
assert torch.unique(torch.cat((state["head_ids"], state["tail_ids"]))).numel() == 151936

rows = []
for line in monitor_path.read_text().splitlines():
    if re.match(r"^\d+,", line):
        parts = [part.strip() for part in line.split(",")]
        rows.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
assert rows, "no GPU monitor samples"
seen = {row[0] for row in rows}
assert seen == set(range(8)), seen
peak_by_gpu = {gpu: max(row[1] for row in rows if row[0] == gpu) for gpu in seen}
print("trainer_global_step=2")
print(f"finite_loss_rows={len(loss_rows)} last_loss={loss_rows[-1]}")
print(f"gate_offset_abs_max={state['gate_offsets'].abs().max().item():.9g}")
print("peak_memory_mib=" + json.dumps(peak_by_gpu, sort_keys=True))
print("PRODUCT_CODE_PRODUCTION_SHAPE_SMOKE_PASS")
PY
)
TASK_SMOKE_RC=$?
set -e
echo "product_code_smoke_exit_code=$TASK_SMOKE_RC"

echo '=== wait for Product Code workers to release all GPUs ==='
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
  echo "ERROR: GPU compute PIDs remain after Product Code test: ${TASK_AFTER_TEST_PIDS[*]}" >&2
  for TASK_PID in "${TASK_AFTER_TEST_PIDS[@]}"; do
    tr '\0' ' ' < "/proc/$TASK_PID/cmdline" || true
    echo
  done
  exit 1
fi
nvidia-smi
echo 'ALL_EIGHT_GPUS_FREE_AFTER_PRODUCT_CODE_SMOKE'

echo '=== restart the verified runner burn on all eight free GPUs ==='
test -x /usr/bin/python3
echo "$TASK_BURN_SHA  $TASK_BURN" | sha256sum -c -
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  nohup /usr/bin/python3 "$TASK_BURN" >"$TASK_BURN_LOG" 2>&1 &
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
  TASK_COUNT="$(grep -oF "$TASK_MARKER" "$TASK_BURN_LOG" | wc -l || true)"
  test "$TASK_COUNT" -eq 8
done
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader
echo "new_burn_launcher=$TASK_NEW_BURN_LAUNCHER workers=${TASK_NEW_BURN_PIDS[*]}"
echo 'CORRECT_ALL_EIGHT_GPU_BURN_RESTARTED_AFTER_PRODUCT_CODE_SMOKE'

test "$TASK_SMOKE_RC" -eq 0
echo 'TH2 PRODUCT CODE PRODUCTION-SHAPE TEST COMPLETE'
