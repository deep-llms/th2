#!/usr/bin/env bash
# Run only after the operator has made all eight GPUs available.
set -euo pipefail
TASK_READY=${1:?Supply the completed CPU readiness directory}
TASK_ROOT=${2:?Supply a fresh experiment output directory}
TASK_PYTHON=/mnt/local/conda-py311/envs/pcc_joint/bin/python3.11
test -x "$TASK_PYTHON"
test -s "$TASK_READY/cpu_ready.json"
test ! -e "$TASK_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=8
"$TASK_PYTHON" scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
# Keep every arm and the final report on one immutable source snapshot even
# when commands.sh is later changed to retrieve logs.
"$TASK_PYTHON" - "$TASK_ROOT" "$TASK_READY" <<'PY'
import json, shutil, sys
from pathlib import Path
root, ready = map(Path, sys.argv[1:])
assert json.loads((ready / 'cpu_ready.json').read_text())['status'] == 'ok'
root.mkdir(parents=True, exist_ok=False)
source = root / 'source'
source.mkdir()
for name in ('pcc', 'scripts', 'tests', 'resources'):
    shutil.copytree(name, source / name, ignore=shutil.ignore_patterns('__pycache__'))
for name in ('run_experiments.py', 'prepare_data.py', 'train.py'):
    shutil.copy2(name, source / name)
shutil.copy2(ready / 'config.json', root / 'config.json')
PY
TASK_ROOT=$(realpath "$TASK_ROOT")
TASK_READY=$(realpath "$TASK_READY")
cd "$TASK_ROOT/source"
for TASK_ARM in Base Shallow Deep; do
  TASK_EXTRA=()
  if [[ "$TASK_ARM" == Deep ]]; then TASK_EXTRA=(--resume-check); fi
  "$TASK_PYTHON" -u -m pcc.joint capacity --config "$TASK_ROOT/config.json" \
    --data-dir "$TASK_READY/inputs" --arm "$TASK_ARM" --physical-gpus 0 1 2 3 4 5 6 7 \
    --output "$TASK_ROOT/capacity-$TASK_ARM" "${TASK_EXTRA[@]}"
done
"$TASK_PYTHON" - "$TASK_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for arm in ('Base', 'Shallow', 'Deep'):
    report = json.loads((root / f'capacity-{arm}/capacity.json').read_text())
    assert report['status'] == 'ok' and report['updates'] == 2 and report['arm'] == arm
    assert report['initial_native_equivalence'] and report['checkpoint_roundtrip']
    assert all(report['parameters_changed'].values())
    assert report['world_size'] == 8 and report['replicas_sha256']
    if arm == 'Deep':
        assert report['resume_next_update_verified']
(root / 'capacity_ready.json').write_text(json.dumps({'status': 'ok', 'arms': ['Base', 'Shallow', 'Deep']}) + '\n')
PY
"$TASK_PYTHON" -m pcc.joint manifest --config "$TASK_ROOT/config.json" \
  --data-dir "$TASK_READY/inputs" --physical-gpus 0 1 2 3 4 5 6 7 --output "$TASK_ROOT/jobs.json"
echo SIX_RUN_QUEUE_START
"$TASK_PYTHON" -u run_experiments.py --config "$TASK_ROOT/jobs.json" \
  --project-dir "$PWD" --run-dir "$TASK_ROOT/runs"
echo SIX_RUN_QUEUE_COMPLETE
