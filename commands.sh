#1 +60+a
#th2-swt-six-checkpoint-corrected-bf16-smoke-20260909-b01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-oish-worker-0
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt_eval
test "$(command -v python)" = /mnt/local/conda-py311/envs/swt_eval/bin/python
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_DEVICE_ORDER=PCI_BUS_ID
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/swt/eval_finetune_smoke_20260909_b01
test ! -e "$TASK_ROOT"
mkdir -p "$TASK_ROOT"
exec > >(tee "$TASK_ROOT/pipeline.log") 2>&1
date -u
hostname
python - <<'PY'
import sys, importlib.metadata as md
print('PYTHON',sys.executable)
for name, expected in [('transformers','5.9.0'),('datasets','4.8.5'),('accelerate','1.13.0'),('lm_eval','0.4.10')]:
    got=md.version(name)
    print(name,got)
    assert got==expected,(name,got,expected)
import torch
print('TORCH',torch.__version__,torch.version.cuda)
assert torch.cuda.is_available() and torch.cuda.device_count()==8
print('ENVIRONMENT_IMPORTS_VERIFIED')
PY
python scripts/verify_manifest.py verify --root . --manifest resources/swt_qwen_launch_5k_20260908.json
python scripts/verify_manifest.py verify --root . --manifest resources/eval_precision_fix_20260909.json
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
TASK_ACCELERATE_TARGET=$(python -c 'from accelerate.commands.config.config_args import default_yaml_config_file; print(default_yaml_config_file)')
mkdir -p "$(dirname "$TASK_ACCELERATE_TARGET")"
cp resources/accelerate_config.yaml "$TASK_ACCELERATE_TARGET"
cmp resources/accelerate_config.yaml "$TASK_ACCELERATE_TARGET"
echo ACCELERATE_CONFIG_COPIED_NO_DDP_LAUNCH_FOR_SINGLE_GPU_TESTS
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_evaluation_pipeline.py -v > "$TASK_ROOT/cpu_tests.log" 2>&1 || {
    tail -n 100 "$TASK_ROOT/cpu_tests.log"
    exit 1
}
tail -n 8 "$TASK_ROOT/cpu_tests.log"
sleep 30
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.check_eval_precision --device cuda --output "$TASK_ROOT/precision.json" > "$TASK_ROOT/precision.log" 2>&1
echo CUDA_PRECISION_GATE_PASSED
test -s "$TASK_ROOT/precision.json"
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
TASK_CHECKPOINT_ROOT=/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_5k_s42_20260908_a01
TASK_ARMS=(B0 A128 A256 A512 C D)
TASK_PIDS=()
for TASK_GPU in 0 1 2 3 4 5; do
    TASK_ARM=${TASK_ARMS[$TASK_GPU]}
    CUDA_VISIBLE_DEVICES="$TASK_GPU" python -u -m scripts.smoke_eval_finetune \
        --checkpoint "$TASK_CHECKPOINT_ROOT/$TASK_ARM/checkpoint-5000" \
        --dataset-root /mnt/local/_data/deep-llms_th2/benchmarks/hf \
        --output-dir "$TASK_ROOT/$TASK_ARM" > "$TASK_ROOT/$TASK_ARM.log" 2>&1 &
    TASK_PIDS+=("$!")
    echo "SMOKE_STARTED arm=$TASK_ARM physical_gpu=$TASK_GPU pid=$!"
done
for TASK_GPU in 0 1 2 3 4 5; do
    TASK_EXIT=0
    wait "${TASK_PIDS[$TASK_GPU]}" || TASK_EXIT=$?
    echo "SMOKE_WORKER_EXIT arm=${TASK_ARMS[$TASK_GPU]} code=$TASK_EXIT"
    printf '%s %s\n' "${TASK_ARMS[$TASK_GPU]}" "$TASK_EXIT" >> "$TASK_ROOT/worker_exits.txt"
    tail -n 5 "$TASK_ROOT/${TASK_ARMS[$TASK_GPU]}.log"
done
sleep 30
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
python - "$TASK_ROOT" <<'PY'
import json, sys
from pathlib import Path
from capacity_allocation.data import write_json
root=Path(sys.argv[1])
precision=json.loads((root/'precision.json').read_text())
reports={}
exits=dict(line.split() for line in (root/'worker_exits.txt').read_text().splitlines())
assert set(exits)=={'B0','A128','A256','A512','C','D'}
for arm in ('B0','A128','A256','A512','C','D'):
    path=root/arm/'report.json'
    if not path.is_file():
        reports[arm]={'success':False,'error':'Missing report; inspect worker log'}
        continue
    result=json.loads(path.read_text())
    assert result['smoke_only'] and result['source_checkpoint_unchanged']
    assert set(result['tasks'])=={'hellaswag','arc_easy','xnli_en'}
    reports[arm]={'success':result['success'] and exits[arm]=='0','worker_exit':int(exits[arm]),'source_checkpoint_unchanged':True,
        'ppl':result['synthetic_long_context_ppl']['token_weighted'],
        'tasks':{name:{'steps':v['training']['steps'],'benchmark_dtypes':v['benchmark_dtypes'],
            'after_benchmark_dtypes':v['after_benchmark_dtypes'],
            'training_dtypes':v['training_dtypes'],'weights_changed':v['weights_changed'],
            'reload_ok':v['reload_ok']} for name,v in result['tasks'].items()}}
summary=dict(success=precision['success'] and all(v['success'] for v in reports.values()),
    smoke_only=True,all_gpus_free=True,precision=precision,checkpoints=reports)
write_json(root/'summary.json',summary)
print(json.dumps(summary,indent=2))
print('SWT_SMOKE_TESTS_FINISHED_ALL_GPUS_FREE_NO_BURNS_STARTED')
if not summary['success']: raise SystemExit(1)
PY

sha256sum "$TASK_ROOT/summary.json"
echo SWT_CORRECTED_BF16_SMOKE_SUCCESS
