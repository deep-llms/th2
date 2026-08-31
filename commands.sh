#1 +60+a
#th2-audit-product-code-prodshape-smoke-and-burn-20260831-a01
set -euo pipefail

TASK_PROJECT=/mnt/local/@PROJECT@
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_OUTPUT=/mnt/local/_outputs/@PROJECT@/product_code_prodshape_smoke_20260831_a01
TASK_CHECKPOINT="$TASK_OUTPUT/checkpoint-2"
TASK_BURN=/tmp/llm_pretrain_burn.py
TASK_BURN_SHA=2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174
TASK_BURN_LOG=/tmp/llm_pretrain_burn_after_product_code_smoke.log

cd "$TASK_PROJECT"
date -u
hostname
pwd
echo 'b17de9f5ab2211c0825922bfaf8f23ed9ecf98c7cae29cc3973074c5dac843a1  compositional/product_code.py' | sha256sum -c -
echo '58a4dfdbe46d45ce674a1f6d3cd8501941b0e8eccda2dc1a047eaeda3fa4bf35  train_compositional.py' | sha256sum -c -

echo '=== validate successful production-shape training artifacts ==='
test -x "$TASK_PYTHON"
test -s "$TASK_OUTPUT/train.log"
test -s "$TASK_OUTPUT/gpu_monitor.log"
test -s "$TASK_OUTPUT/train_config.json"
test -s "$TASK_OUTPUT/train_results.json"
test -s "$TASK_OUTPUT/embedding.pt"
test ! -e "$TASK_OUTPUT/output_head.pt"
test ! -e "$TASK_CHECKPOINT/output_head.pt"
for TASK_REQUIRED in config.json model.safetensors trainer_state.json optimizer.pt scheduler.pt \
    embedding.pt rng_state_0.pth rng_state_1.pth rng_state_2.pth rng_state_3.pth \
    rng_state_4.pth rng_state_5.pth rng_state_6.pth rng_state_7.pth; do
  test -s "$TASK_CHECKPOINT/$TASK_REQUIRED"
done
if grep -E -i 'Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError:|NCCL[^[:cntrl:]]*(unhandled|system error|remote process exited|watchdog|collective operation timeout)' "$TASK_OUTPUT/train.log"; then
  echo 'ERROR: fatal signature found in training log' >&2
  exit 1
fi

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
"$TASK_PYTHON" - "$TASK_CHECKPOINT" "$TASK_OUTPUT/train_config.json" \
  "$TASK_OUTPUT/train_results.json" "$TASK_OUTPUT/gpu_monitor.log" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

import torch

from compositional.loading import _build_arm_from_config
from compositional.tied_head import make_tied_head

checkpoint = Path(sys.argv[1])
config_path = Path(sys.argv[2])
results_path = Path(sys.argv[3])
monitor_path = Path(sys.argv[4])

trainer = json.loads((checkpoint / "trainer_state.json").read_text())
assert trainer["global_step"] == 2, trainer["global_step"]
loss_rows = [row for row in trainer["log_history"] if "loss" in row]
assert len(loss_rows) >= 2, loss_rows
for row in loss_rows:
    for key in ("loss", "grad_norm", "learning_rate"):
        assert math.isfinite(float(row[key])), (key, row[key])
results = json.loads(results_path.read_text())
for key, value in results.items():
    if isinstance(value, (int, float)):
        assert math.isfinite(float(value)), (key, value)

config = json.loads(config_path.read_text())["compositional"]
assert config["arm"] == "product_code"
assert config["tie_output"] is True
assert config["independent_lowrank_output"] is False
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
assert tuple(state["bias"].shape) == (1024,)
assert torch.isfinite(state["E_h"]).all()
assert all(torch.isfinite(state[f"C.{index}"]).all() for index in range(4))
assert torch.isfinite(state["gate_offsets"]).all()
assert state["gate_offsets"].abs().max().item() > 0.0
assert state["codes"].min().item() >= 0
assert state["codes"].max().item() < 4096
all_ids = torch.cat((state["head_ids"], state["tail_ids"]))
assert torch.unique(all_ids).numel() == 151936
assert torch.equal(torch.sort(all_ids).values, torch.arange(151936))

module = _build_arm_from_config(config, 151936, 1024, state=state)
module.load_state_dict(state, strict=True)
module.eval()
assert module.parameter_count == 19_474_944
probe_ids = torch.stack((
    module.head_ids[:4],
    module.tail_ids[:4],
)).reshape(-1)
with torch.no_grad():
    probe = module.materialize(probe_ids)
    assert tuple(probe.shape) == (8, 1024)
    assert torch.isfinite(probe).all()
    hidden = torch.randn(1, 2, 1024)
    tied_head = make_tied_head(module, "product_code", 151936)
    logits = tied_head(hidden)
    reference = hidden @ module.materialize().T
    torch.testing.assert_close(logits, reference, rtol=0, atol=0)
    assert tuple(logits.shape) == (1, 2, 151936)

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
print("trainer_global_step=2")
print(f"finite_loss_rows={len(loss_rows)} last_loss={loss_rows[-1]}")
print(f"train_results={results}")
print(f"gate_offset_abs_max={state['gate_offsets'].abs().max().item():.9g}")
print("peak_memory_mib=" + json.dumps(peak_by_gpu, sort_keys=True))
print("PRODUCT_CODE_CHECKPOINT_AND_EXACT_TIED_HEAD_AUDIT_PASS")
PY

echo '=== validate restored correct communicating burn on all eight GPUs ==='
echo "$TASK_BURN_SHA  $TASK_BURN" | sha256sum -c -
test -s "$TASK_BURN_LOG"
mapfile -t TASK_GPU_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d;s/[[:space:]]//g' | sort -nu
)
test "${#TASK_GPU_PIDS[@]}" -eq 8
TASK_LAUNCHER=''
for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
  test "$TASK_PID" != 1
  TASK_PPID="$(awk '/^PPid:/ {print $2}' "/proc/$TASK_PID/status")"
  test "$TASK_PPID" != 1
  if [[ -z "$TASK_LAUNCHER" ]]; then
    TASK_LAUNCHER="$TASK_PPID"
  else
    test "$TASK_PPID" = "$TASK_LAUNCHER"
  fi
done
TASK_LAUNCHER_CMD="$(tr '\0' ' ' < "/proc/$TASK_LAUNCHER/cmdline")"
[[ "$TASK_LAUNCHER_CMD" == *"$TASK_BURN"* ]]
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
rows = []
for line in sys.argv[1].splitlines():
    fields = [field.strip() for field in line.split(",")]
    rows.append((int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])))
assert len(rows) == 8, rows
assert {row[0] for row in rows} == set(range(8))
for gpu, used, total, util in rows:
    assert used >= 140_000, (gpu, used, total)
    assert used < total, (gpu, used, total)
print("ALL_EIGHT_RESTORED_BURNS_USE_MOST_GPU_MEMORY")
PY
echo "burn_launcher=$TASK_LAUNCHER gpu_workers=${TASK_GPU_PIDS[*]}"
echo 'TH2 PRODUCT CODE PRODUCTION-SHAPE TRAINING AND BURN AUDIT PASS'
