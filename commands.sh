#1 +60+a
#th2-readonly-check-ranklift-10k-completion-20260902-a01
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_OUTPUT="$TASK_OUTPUT_BASE/ranklift_tied_c124_m460"
TASK_CKPT="$TASK_OUTPUT/checkpoint-10000"
TASK_LOG="$TASK_OUTPUT_BASE/logs/ranklift_tied_c124_m460_10k_20260902/ranklift_tied_c124_m460.log"
TASK_EXPERIMENT_LOG="$TASK_OUTPUT_BASE/logs/ranklift_tied_c124_m460_10k_20260902/experiments.log"

echo '=== current GPU state ==='
date -u
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits || true

echo '=== relevant live processes ==='
pgrep -af 'run_ranklift_b200_10k.py|run_experiments.py|train_compositional.py|/tmp/llm_pretrain_burn.py' || true

echo '=== checkpoint validation ==='
if [[ -d "$TASK_CKPT" ]]; then
    /mnt/local/conda-py311/envs/sparse_emb/bin/python3.11 - "$TASK_CKPT" <<'PY'
import json
import math
from pathlib import Path
import sys

checkpoint = Path(sys.argv[1])
required = (
    "config.json", "model.safetensors", "trainer_state.json",
    "optimizer.pt", "scheduler.pt", "embedding.pt",
    *(f"rng_state_{rank}.pth" for rank in range(8)),
)
missing = [name for name in required if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0]
print(f"missing={missing}")
state = json.loads((checkpoint / "trainer_state.json").read_text())
losses = [float(row["loss"]) for row in state.get("log_history", []) if "loss" in row]
print(f"global_step={state['global_step']} logged_losses={len(losses)} first_loss={losses[0] if losses else None} last_loss={losses[-1] if losses else None}")
assert int(state["global_step"]) == 10000
assert not missing
assert losses and all(math.isfinite(value) for value in losses)
print("CHECKPOINT_10000_COMPLETE_AND_VALID")
PY
else
    echo 'checkpoint-10000: pending'
    find "$TASK_OUTPUT" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 5 || true
fi

echo '=== logs ==='
for TASK_FILE in "$TASK_EXPERIMENT_LOG" "$TASK_LOG"; do
    echo "--- $TASK_FILE ---"
    if [[ -s "$TASK_FILE" ]]; then
        tr '\r' '\n' < "$TASK_FILE" | tail -n 100
    else
        echo 'missing or empty'
    fi
done

echo '=== narrow fatal scan ==='
if grep -HniE 'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|ChildFailedError|ProcessExitedException|Segmentation fault|Bus error' "$TASK_LOG" "$TASK_EXPERIMENT_LOG" 2>/dev/null; then
    echo 'FATAL_SIGNATURE_FOUND' >&2
    exit 1
fi
echo 'NO_FATAL_SIGNATURE_FOUND'
echo 'TH2 READONLY RANKLIFT COMPLETION CHECK FINISHED; PROCESSES UNMODIFIED'
