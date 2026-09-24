#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${1:?Fresh student experiment root}
TASK_JOINT=${2:?Completed joint experiment root}
TASK_INPUTS=${3:?Verified existing input directory}
TASK_INSPECTION=${4:?Fresh GPU ownership inspection JSON}
TASK_CPU_READY=${5:?Passing CPU regression receipt for this source version}
TASK_PYTHON=/mnt/local/conda-py311/envs/pcc_joint/bin/python3.11
test -x "$TASK_PYTHON"
test -s "$TASK_JOINT/runs/report/complete.json"
test -s "$TASK_INPUTS/complete.json"
test ! -e "$TASK_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=8
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" - "$TASK_CPU_READY" <<'PY'
import json,sys
from pathlib import Path
from pcc.joint_training import code_identity
ready=json.loads(Path(sys.argv[1]).read_text())
assert ready['status']=='ok' and ready['code']==code_identity(), 'CPU readiness/source mismatch'
print('DISTILL_CPU_SOURCE_VERIFIED',flush=True)
PY
# This script is called only inside with_gpu_guard_disabled, after CPU checks.
PYTHONPATH="$PWD" /usr/bin/python3 -u -m scripts.verified_gpu_reclaim stop \
  --authorized-stop --inspection "$TASK_INSPECTION" --output "${TASK_ROOT}-gpu-reclaim.json"
"$TASK_PYTHON" - "$TASK_ROOT" "$TASK_JOINT" <<'PY'
import json,shutil,sys
from pathlib import Path
root,joint=map(Path,sys.argv[1:])
assert json.loads((joint/'runs/report/complete.json').read_text())['decision']=='recommend_distillation_design'
root.mkdir(parents=True,exist_ok=False)
source=root/'source';source.mkdir()
for name in ('pcc','scripts','tests','resources'):
    shutil.copytree(name,source/name,ignore=shutil.ignore_patterns('__pycache__'))
for name in ('run_experiments.py','prepare_data.py','train.py'):
    shutil.copy2(name,source/name)
shutil.copy2(joint/'config.json',root/'config.json')
PY
TASK_ROOT=$(realpath "$TASK_ROOT")
TASK_JOINT=$(realpath "$TASK_JOINT")
TASK_INPUTS=$(realpath "$TASK_INPUTS")
cd "$TASK_ROOT/source"
for TASK_SEED in 0 1; do
  "$TASK_PYTHON" -u -m pcc.distill audit --config "$TASK_ROOT/config.json" \
    --joint-root "$TASK_JOINT" --data-dir "$TASK_INPUTS" --seed-index "$TASK_SEED" \
    --physical-gpus 0 1 2 3 4 5 6 7 --output "$TASK_ROOT/audit-seed-$TASK_SEED"
done
"$TASK_PYTHON" - "$TASK_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for seed in (0,1):
    record=json.loads((root/f'audit-seed-{seed}/complete.json').read_text())
    if not (record['status']=='ok' and record['ready_for_students']):
        raise SystemExit('SCIENTIFIC_STOP: no usable same-checkpoint feedback advantage; no student training')
print('BOTH_TEACHER_AUDITS_PASSED',flush=True)
PY
for TASK_ARM in LM PCC; do
  "$TASK_PYTHON" -u -m pcc.distill capacity --config "$TASK_ROOT/config.json" \
    --joint-root "$TASK_JOINT" --data-dir "$TASK_INPUTS" --seed-index 0 --arm "$TASK_ARM" \
    --audit-dir "$TASK_ROOT/audit-seed-0" --physical-gpus 0 1 2 3 4 5 6 7 \
    --output "$TASK_ROOT/capacity-$TASK_ARM"
done
"$TASK_PYTHON" - "$TASK_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for arm in ('LM','PCC'):
    record=json.loads((root/f'capacity-{arm}/capacity.json').read_text())
    assert record['status']=='ok' and record['world_size']==8
    assert record['teacher_unchanged'] and record['student_changed'] and record['checkpoint_roundtrip']
    assert record['resume_next_update_verified'] and record['student_noop_exact']
(root/'capacity_ready.json').write_text(json.dumps({'status':'ok','arms':['LM','PCC']})+'\n')
PY
"$TASK_PYTHON" -m pcc.distill manifest --config "$TASK_ROOT/config.json" \
  --joint-root "$TASK_JOINT" --data-dir "$TASK_INPUTS" --audit-root "$TASK_ROOT" --output "$TASK_ROOT/jobs.json"
echo FOUR_STUDENT_QUEUE_START
"$TASK_PYTHON" -u run_experiments.py --config "$TASK_ROOT/jobs.json" --project-dir "$PWD" --run-dir "$TASK_ROOT/runs"
echo FOUR_STUDENT_QUEUE_COMPLETE
