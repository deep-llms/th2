#1 +90+a
#th2-readonly-inventory-checkpoints-after-pod-return-20260902-a02
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11

echo '=== machine and GPUs ==='
date -u
hostname
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits || true

echo '=== storage ==='
df -h /mnt/local
du -sh "$TASK_OUTPUT_BASE" 2>/dev/null || true

echo '=== complete checkpoint inventory and structural validation ==='
test -x "$TASK_PYTHON"
test -d "$TASK_OUTPUT_BASE"
"$TASK_PYTHON" - "$TASK_OUTPUT_BASE" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
checkpoints = sorted(
    root.glob("*/checkpoint-*"),
    key=lambda path: (path.parent.name, int(path.name.removeprefix("checkpoint-"))),
)
print(f"output_root={root} checkpoint_count={len(checkpoints)}")
assert checkpoints, "no checkpoints found"
bad = []
for checkpoint in checkpoints:
    step_text = checkpoint.name.removeprefix("checkpoint-")
    # Dense Qwen checkpoints intentionally have no separate embedding.pt;
    # require only artifacts common to dense and compositional checkpoints.
    required = ("config.json", "model.safetensors", "trainer_state.json")
    missing = [
        name for name in required
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0
    ]
    state_step = None
    try:
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        state_step = int(state["global_step"])
    except Exception as error:
        missing.append(f"invalid trainer_state.json: {error}")
    if state_step is not None and state_step != int(step_text):
        missing.append(f"trainer_state_step={state_step}")
    status = "OK" if not missing else f"BAD {missing}"
    print(f"{checkpoint.relative_to(root)} | {status}")
    if missing:
        bad.append((str(checkpoint), missing))
print(f"checkpoint_count={len(checkpoints)} invalid_count={len(bad)}")
assert not bad, bad
print("ALL_DISCOVERED_CHECKPOINTS_STRUCTURALLY_VALID")
PY

echo '=== latest checkpoint per experiment ==='
for TASK_DIR in "$TASK_OUTPUT_BASE"/*; do
    [[ -d "$TASK_DIR" ]] || continue
    TASK_LATEST="$(find "$TASK_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)"
    [[ -n "$TASK_LATEST" ]] && printf '%s | %s\n' "$(basename "$TASK_DIR")" "$TASK_LATEST"
done

echo '=== final-interface smoke and RankLift state ==='
find "$TASK_OUTPUT_BASE/final_interfaces_smoke_20260902" -maxdepth 3 -type f \
    -printf '%s %p\n' 2>/dev/null | sort || true
find "$TASK_OUTPUT_BASE/ranklift_tied_c124_m460" -maxdepth 1 -type d \
    -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V || true

echo 'TH2 READONLY CHECKPOINT INVENTORY COMPLETE; NOTHING MODIFIED'
