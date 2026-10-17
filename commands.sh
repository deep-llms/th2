#1 +60+a
#th2-q359-template-smoke-20261017-a02
set -euo pipefail
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_PYTHON=/mnt/local/conda-py311/envs/eval_fa4/bin/python
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-template-smoke-20261017-a02
TASK_SESSION=template_smoke_20261017_a02
test ! -e "$TASK_ROOT"
mkdir "$TASK_ROOT"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1
"$TASK_PYTHON" - "$TASK_ROOT" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
from accelerate.commands.config.config_args import default_config_file
from scripts.verified_gpu_reclaim import inspect
root=Path(sys.argv[1]); source=Path('template_validation')
for name,expected in json.loads((source/'source_manifest.json').read_text()).items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==expected,name
inspection=inspect()
assert not inspection['guard_disabled'],'GPU owner guard held'
(root/'inspection.json').write_text(json.dumps(inspection,indent=2))
target=Path(default_config_file); target.parent.mkdir(parents=True,exist_ok=True)
if target.exists(): shutil.copy2(target,root/'previous_accelerate.yaml')
shutil.copy2('resources/accelerate_config.yaml',target)
assert target.read_bytes()==Path('resources/accelerate_config.yaml').read_bytes()
print('ACCELERATE_CONFIG_COPIED',target)
PY
"$TASK_PYTHON" -m accelerate.commands.env | tee "$TASK_ROOT/accelerate-env.txt"
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" -c 'import sys; sys.path.insert(0,"template_validation"); import train; from importlib.metadata import version; assert version("flash-attn-4")=="4.0.0b33"; print("TEMPLATE_IMPORT_OK")'
"$TASK_PYTHON" run_experiments.py --config template_validation/jobs.b200.json --list
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
printf -v TASK_CMD 'exec %q -B -m scripts.train_then_burn --config %q --output %q --inspection %q --host thiennh-p6-q359-worker-0 --burn-session %q >%q 2>&1' "$TASK_PYTHON" "$PWD/template_validation/jobs.b200.json" "$TASK_ROOT/supervised" "$TASK_ROOT/inspection.json" "${TASK_SESSION}_final_burn" "$TASK_ROOT/supervisor.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 5
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0
cat "$TASK_ROOT/supervisor.log"
echo TEMPLATE_SMOKE_SUPERVISOR_STARTED
