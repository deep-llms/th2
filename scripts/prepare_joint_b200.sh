#!/usr/bin/env bash
# CPU-only deployment validation and one shared input cache.
set -euo pipefail
TASK_ROOT=${1:?Supply a fresh output directory}
TASK_PYTHON=/mnt/local/conda-py311/envs/pcc_joint/bin/python3.11
test -x "$TASK_PYTHON"
test ! -e "$TASK_ROOT"
mkdir -p "$TASK_ROOT"
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=8
cp pcc.joint.b200.json "$TASK_ROOT/config.json"
"$TASK_PYTHON" -m pip check
"$TASK_PYTHON" - "$TASK_ROOT/runtime.json" <<'PY'
import importlib.metadata, json, platform, sys
from pathlib import Path
packages = ('torch', 'transformers', 'numpy', 'datasets', 'matplotlib', 'accelerate')
record = {p: importlib.metadata.version(p) for p in packages}
record.update(host=platform.node(), python=sys.executable)
assert record['transformers'] == '4.57.1', record
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
PY
"$TASK_PYTHON" scripts/verify_manifest.py verify \
  --root /mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base-ddc928429ed09d9ad603fd762053d0434c15e865 \
  --manifest resources/qwen3_joint_assets.json
echo CPU_REGRESSION_START
if ! "$TASK_PYTHON" -u -m unittest discover -s tests -v > "$TASK_ROOT/tests.log" 2>&1; then
  tail -n 100 "$TASK_ROOT/tests.log"
  exit 1
fi
tail -n 5 "$TASK_ROOT/tests.log"
echo INPUT_PREPARATION_START
if ! "$TASK_PYTHON" -u -m pcc.joint prepare --config "$TASK_ROOT/config.json" \
  --output "$TASK_ROOT/inputs" > "$TASK_ROOT/prepare.log" 2>&1; then
  tail -n 100 "$TASK_ROOT/prepare.log"
  exit 1
fi
"$TASK_PYTHON" - "$TASK_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
inputs = json.loads((root / 'inputs/complete.json').read_text())
assert inputs['status'] == 'ok'
record = {'status': 'ok', 'gpu_used': False, 'inputs': str(root / 'inputs'),
          'runtime': json.loads((root / 'runtime.json').read_text()),
          'streams': inputs['streams']}
(root / 'cpu_ready.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
PY
echo B200_CPU_READY
