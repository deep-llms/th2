#!/usr/bin/env bash
# English six-arm/seven-checkpoint sweep. No downloads, signals, cleanup or burns.
set -euo pipefail
TASK_PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$TASK_PROJECT_DIR"
TASK_RUN_ROOT=${1:?Pass a fresh absolute output directory}
[[ "$TASK_RUN_ROOT" == /mnt/local/_outputs/deep-llms_th2/swt/* ]]
[[ ! -e "$TASK_RUN_ROOT" && ! -L "$TASK_RUN_ROOT" ]]
test "$(hostname)" = thiennh-p6-oish-worker-0
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt_eval
test "$(command -v python)" = /mnt/local/conda-py311/envs/swt_eval/bin/python
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_DEVICE_ORDER=PCI_BUS_ID
TASK_TRAIN_ROOT=/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_5k_s42_20260908_a01
TASK_BENCHMARKS=/mnt/local/_data/deep-llms_th2/benchmarks/hf
TASK_EVAL_DATA=/mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B/eval
TASK_TOKENIZER=/mnt/local/_models/deep-llms_th2/Qwen3-0.6B
TASK_CACHE=/mnt/local/_data/deep-llms_th2/swt/qwen_en_map160_batch1000
TASK_GPUS=(0 1 2 3 4 5 6 7)
TASK_CHECKPOINTS=()
for TASK_ARM in B0 A128 A256 A512 C D; do
    for TASK_STEP in 250 500 1000 2000 3000 4000 5000; do
        TASK_CHECKPOINTS+=("${TASK_ARM}_step${TASK_STEP}=$TASK_TRAIN_ROOT/$TASK_ARM/checkpoint-$TASK_STEP")
    done
done
test "${#TASK_CHECKPOINTS[@]}" = 42
python scripts/gpu_status.py --gpus "${TASK_GPUS[@]}" --require-free
mkdir "$TASK_RUN_ROOT"
exec > >(tee "$TASK_RUN_ROOT/pipeline.log") 2>&1
trap 'TASK_EXIT=$?; echo "SWT_EVAL_FINETUNE_FAILED exit=$TASK_EXIT; subsequent stages not started" >&2' ERR
date -u
python scripts/verify_manifest.py verify --root "$TASK_PROJECT_DIR" \
    --manifest resources/swt_qwen_launch_5k_20260908.json
python scripts/verify_manifest.py verify --root "$TASK_BENCHMARKS" \
    --manifest resources/english_core_benchmark_files_20260908.json

# CPU-only preparation: validate full splits and warm the common packing cache once.
CUDA_VISIBLE_DEVICES='' python -u - "$TASK_TRAIN_ROOT" "$TASK_BENCHMARKS" "$TASK_EVAL_DATA" "$TASK_TOKENIZER" "$TASK_CACHE" "$TASK_RUN_ROOT" <<'PY'
import json, sys
from pathlib import Path
from eval.runtime import offline, checkpoint_identity
offline()
from eval.benchmarks import task_plan, load_tasks
from capacity_allocation.data import load_text_data, preprocess_text, write_json
from transformers import AutoTokenizer
training, benchmarks, eval_data, tokenizer_path, cache, output = map(Path, sys.argv[1:])
plan, unavailable = task_plan('en')
assert len(plan) == 78 and not unavailable
tasks = load_tasks(plan, benchmarks)
coverage = {}
for name, task in tasks.items():
    docs = task.eval_docs
    assert len(docs) > 0, name
    coverage[name] = dict(eval_examples=len(docs), fingerprint=docs._fingerprint)
for name in ('hellaswag', 'arc_easy', 'xnli_en'):
    assert tasks[name].has_training_docs(), name
    docs = tasks[name].training_docs()
    assert len(docs) > 0, name
    coverage[name]['training_examples'] = len(docs)
    coverage[name]['training_fingerprint'] = docs._fingerprint
identities = {}
for arm in ('B0', 'A128', 'A256', 'A512', 'C', 'D'):
    for step in (250, 500, 1000, 2000, 3000, 4000, 5000):
        checkpoint = training/arm/f'checkpoint-{step}'
        state = json.loads((checkpoint/'trainer_state.json').read_text())
        assert state['global_step'] == step, str(checkpoint)
        identities[f'{arm}_step{step}'] = checkpoint_identity(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
tokenizer.model_max_length = 10**30
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'
packed = preprocess_text(load_text_data(eval_data, ('en',)), tokenizer,
    block_size=2048, num_proc=160, batch_size=1000, cache_dir=cache)
assert len(packed) == 4802 and len(packed)*2047 == 9829694
write_json(output/'inputs_verified.json', dict(success=True, checkpoints=identities,
    benchmark_coverage=coverage, ppl_fingerprint=packed._fingerprint, scored_targets=9829694))
print('ALL_42_CHECKPOINTS_78_TASKS_3_TRAIN_SPLITS_AND_PPL_CACHE_VERIFIED', flush=True)
PY

TASK_COMMON=(--checkpoints "${TASK_CHECKPOINTS[@]}" --dataset-root "$TASK_BENCHMARKS"
    --tokenizer-name "$TASK_TOKENIZER" --languages en --precision bf16
    --benchmark-batch-size 8 --gpus "${TASK_GPUS[@]}")
TASK_EVAL=(python -u -m eval.eval_parallel "${TASK_COMMON[@]}"
    --eval-dir "$TASK_EVAL_DATA" --batch-size 1 --preprocessing-num-workers 160
    --preprocessing-batch-size 1000 --preprocessing-cache-dir "$TASK_CACHE"
    --output-dir "$TASK_RUN_ROOT/eval")
TASK_FINETUNE=(python -u -m finetune.run_all "${TASK_COMMON[@]}"
    --tasks hellaswag arc_easy xnli --seeds 42 123 456 --train-language en
    --output-dir "$TASK_RUN_ROOT/finetune")
"${TASK_EVAL[@]}" --dry-run > "$TASK_RUN_ROOT/eval_plan.json"
"${TASK_FINETUNE[@]}" --dry-run > "$TASK_RUN_ROOT/finetune_plan.json"
python - "$TASK_RUN_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for stage, expected in (('eval', 84), ('finetune', 378)):
    jobs = json.loads((root/f'{stage}_plan.json').read_text())
    assert len(jobs) == expected and len({j['name'] for j in jobs}) == expected
print('PLANS_VERIFIED: 84 eval workers, then 378 independent fine-tunes')
PY

copy_accelerate_config() {
    local TASK_ACCELERATE_TARGET
    TASK_ACCELERATE_TARGET=$(python -c 'from accelerate.commands.config.config_args import default_yaml_config_file; print(default_yaml_config_file)')
    mkdir -p "$(dirname "$TASK_ACCELERATE_TARGET")"
    cp resources/accelerate_config.yaml "$TASK_ACCELERATE_TARGET"
    cmp resources/accelerate_config.yaml "$TASK_ACCELERATE_TARGET"
    python - "$TASK_ACCELERATE_TARGET" <<'PY'
import sys
from accelerate.commands.config.config_args import load_config_from_file
config = load_config_from_file(sys.argv[1])
assert config.num_processes == 8 and config.mixed_precision == 'bf16'
assert config.distributed_type.value == 'MULTI_GPU' and not config.use_cpu
print('ACCELERATE_CONFIG_COPIED_AND_VERIFIED', sys.argv[1])
PY
}
verify_stage() {
    python - "$TASK_RUN_ROOT" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
root, stage, expected = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
result = json.loads((root/stage/'complete.json').read_text())
assert result['success'] is True and len(result['completed']) == expected
assert all(r['exit_code'] == 0 and r['error'] is None for r in result['completed'])
reference = json.loads((root/'inputs_verified.json').read_text())
identities = {r['path']: r for r in reference['checkpoints'].values()}
for record in result['completed']:
    item = json.loads(Path(record['result']).read_text())
    assert item['success'] is True and item['languages'] == ['en']
    assert item['checkpoint'] == identities[item['checkpoint']['path']]
    if 'ppl' in item:
        row = item['ppl']['by_language']['en']
        assert row['scored_targets'] == reference['scored_targets']
        assert row['data_fingerprint'] == reference['ppl_fingerprint']
    for name, benchmark in item.get('benchmarks', {}).items():
        assert benchmark['samples']['effective'] == reference['benchmark_coverage'][name]['eval_examples']
    if stage == 'finetune':
        assert item['precision'] == 'bf16' and item['master_weights'] == 'fp32'
        config = item['config']
        assert config['epochs'] == 3 and config['lr'] == 2e-5 and config['max_length'] == 256
        assert config['batch_size'] == (16 if item['task'] == 'hellaswag' else 32)
        expected_steps = ((item['used_examples'] + config['batch_size'] - 1)//config['batch_size'])*3
        assert item['training']['steps'] == expected_steps
print('STAGE_VERIFIED', stage, expected, flush=True)
PY
}
sleep 30
python scripts/gpu_status.py --gpus "${TASK_GPUS[@]}" --require-free
copy_accelerate_config
sleep 30
python scripts/gpu_status.py --gpus "${TASK_GPUS[@]}" --require-free
echo START_FULL_ENGLISH_EVALUATION
"${TASK_EVAL[@]}"
verify_stage eval 84
sleep 30
python scripts/gpu_status.py --gpus "${TASK_GPUS[@]}" --require-free
copy_accelerate_config
sleep 30
python scripts/gpu_status.py --gpus "${TASK_GPUS[@]}" --require-free
echo START_INDEPENDENT_ENGLISH_FINETUNING
"${TASK_FINETUNE[@]}"
verify_stage finetune 378
sleep 30
python scripts/gpu_status.py --gpus "${TASK_GPUS[@]}" --require-free
python - "$TASK_RUN_ROOT" <<'PY'
import sys
from pathlib import Path
from datetime import datetime, timezone
from capacity_allocation.data import write_json
write_json(Path(sys.argv[1])/'complete.json', dict(success=True,
    checkpoints=42, eval_jobs=84, finetune_jobs=378,
    completed_utc=datetime.now(timezone.utc).isoformat()))
PY
echo SWT_FULL_EVAL_AND_FINETUNE_COMPLETE
