#1 +60+a
#th2-joint-v2-eight-gpu-20260923-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true
TASK_PYTHON=/mnt/local/conda-py311/envs/pcc_joint/bin/python3.11
TASK_READY=/mnt/local/_outputs/@PROJECT@/joint-v2-readiness-20260923-a01
TASK_RUN=/mnt/local/_outputs/@PROJECT@/joint-v2-b200-20260923-a01
date -u
hostname
# The CPU producer is already running. Do not resubmit preprocessing.
for TASK_ATTEMPT in {1..90}; do
  if test -s "$TASK_READY/cpu_ready.json"; then break; fi
  if test -e "$TASK_READY/inputs/failure.json"; then
    cat "$TASK_READY/inputs/failure.json"
    exit 1
  fi
  echo "Waiting for verified shared inputs: attempt $TASK_ATTEMPT/90"
  sleep 10
done
test -s "$TASK_READY/cpu_ready.json"
"$TASK_PYTHON" - "$TASK_READY" <<'PYTHON'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
record = json.loads((root / 'cpu_ready.json').read_text())
assert record['status'] == 'ok'
assert record['streams']['train']['input_tokens'] == 201326592
assert record['streams']['dev']['input_tokens'] == 2000000
assert json.loads((root / 'config.json').read_text()) == json.loads(Path('pcc.joint.b200.json').read_text())
print('LONG_INPUTS_VERIFIED', flush=True)
PYTHON
# Explicit user authorization was given for stopping this verified GPU burn.
"$TASK_PYTHON" -u -m scripts.reclaim_b200_burn_20260923 --authorized-stop   --output /mnt/local/_outputs/@PROJECT@/joint-v2-burn-stop-20260923-a01.json
bash scripts/launch_joint_b200.sh "$TASK_READY" "$TASK_RUN"
