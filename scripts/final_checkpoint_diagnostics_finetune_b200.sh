#!/usr/bin/env bash
# Only step5000: immutable diagnostic inputs -> 18 diagnostic jobs -> 54 fine-tunes.
# Existing zero-shot results/checkpoints/caches are preserved. No signals/downloads.
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-oish-worker-0
TASK_ROOT=${1:?Fresh absolute output root required}
[[ "$TASK_ROOT" == /mnt/local/_outputs/deep-llms_th2/swt/* ]]
test ! -e "$TASK_ROOT"
test ! -L "$TASK_ROOT"
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt_eval
test "$(command -v python)" = /mnt/local/conda-py311/envs/swt_eval/bin/python
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK_TRAIN=/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_5k_s42_20260908_a01
TASK_OLD=/mnt/local/_outputs/deep-llms_th2/swt/full_eval_finetune_42ckpt_20260909_a01
TASK_BUNDLE=/mnt/local/_data/deep-llms_th2/swt/diagnostics/qwen_en_pool_20260909_a01
TASK_DATA=/mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B
TASK_CACHE=/mnt/local/_data/deep-llms_th2/swt/qwen_en_map160_batch1000
TASK_TOKENIZER=/mnt/local/_models/deep-llms_th2/Qwen3-0.6B
TASK_BENCH=/mnt/local/_data/deep-llms_th2/benchmarks/hf
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
mkdir "$TASK_ROOT"
exec > >(tee "$TASK_ROOT/pipeline.log") 2>&1
trap 'echo "FINAL_ONLY_PIPELINE_FAILED: no subsequent stage launched" >&2' ERR
date -u
python scripts/verify_manifest.py verify --root . --manifest resources/swt_qwen_launch_5k_20260908.json
python scripts/verify_manifest.py verify --root "$TASK_BENCH" --manifest resources/english_core_benchmark_files_20260908.json
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_diagnostics.py -v
echo START_FROZEN_DIAGNOSTIC_BUNDLE
CUDA_VISIBLE_DEVICES='' python -u -m eval.diagnostic_data \
    --train-dir "$TASK_DATA/train" --eval-dir "$TASK_DATA/eval" \
    --tokenizer-name "$TASK_TOKENIZER" --languages en --vocab-size 151936 \
    --block-size 2048 --preprocessing-num-workers 160 --preprocessing-batch-size 1000 \
    --preprocessing-cache-dir "$TASK_CACHE" --training-shuffle-seed 42 \
    --probe-blocks 8 --seed 42 --output-dir "$TASK_BUNDLE"
copy_config() {
    local TASK_CONFIG
    TASK_CONFIG=$(python -c 'from accelerate.commands.config.config_args import default_yaml_config_file; print(default_yaml_config_file)')
    mkdir -p "$(dirname "$TASK_CONFIG")"
    cp resources/accelerate_config.yaml "$TASK_CONFIG"
    cmp resources/accelerate_config.yaml "$TASK_CONFIG"
    python - "$TASK_CONFIG" <<'PY'
import sys
from accelerate.commands.config.config_args import load_config_from_file
c = load_config_from_file(sys.argv[1])
assert c.num_processes == 8 and c.mixed_precision == 'bf16'
assert c.distributed_type.value == 'MULTI_GPU' and not c.use_cpu
print('ACCELERATE_CONFIG_VERIFIED', sys.argv[1])
PY
}
sleep 30
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
copy_config
sleep 30
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
TASK_DIAG=(python -u -m scripts.final_checkpoint_diagnostics --run-root "$TASK_ROOT"
    --training-root "$TASK_TRAIN" --bundle "$TASK_BUNDLE" --tokenizer "$TASK_TOKENIZER" --old-run "$TASK_OLD")
echo START_STEP5000_DIAGNOSTICS
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.smoke_final_diagnostics \
    --training-root "$TASK_TRAIN" --bundle "$TASK_BUNDLE" --tokenizer "$TASK_TOKENIZER" \
    --output "$TASK_ROOT/diagnostic_smoke.json"
sleep 30
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
"${TASK_DIAG[@]}"
sleep 30
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
copy_config
sleep 30
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
TASK_CHECKPOINTS=()
for TASK_ARM in B0 A128 A256 A512 C D; do
    TASK_CHECKPOINTS+=("${TASK_ARM}_step5000=$TASK_TRAIN/$TASK_ARM/checkpoint-5000")
done
TASK_FT=(python -u -m finetune.run_all --checkpoints "${TASK_CHECKPOINTS[@]}"
    --dataset-root "$TASK_BENCH" --tokenizer-name "$TASK_TOKENIZER" --languages en
    --precision bf16 --benchmark-batch-size 8 --gpus 0 1 2 3 4 5 6 7
    --tasks hellaswag arc_easy xnli --seeds 42 123 456 --train-language en
    --output-dir "$TASK_ROOT/finetune")
"${TASK_FT[@]}" --dry-run > "$TASK_ROOT/finetune_plan.json"
python - "$TASK_ROOT/finetune_plan.json" <<'PY'
import json, sys
from pathlib import Path
jobs = json.loads(Path(sys.argv[1]).read_text())
assert len(jobs) == 54 and len({j['name'] for j in jobs}) == 54
assert all(Path(j['checkpoint']).name == 'checkpoint-5000' for j in jobs)
print('FINAL_ONLY_FINETUNE_PLAN_VERIFIED', len(jobs))
PY
echo START_STEP5000_FINETUNING
"${TASK_FT[@]}"
"${TASK_DIAG[@]}" --verify-finetune
sleep 30
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
python - "$TASK_ROOT" <<'PY'
from pathlib import Path
import sys
from datetime import datetime, timezone
from capacity_allocation.data import write_json
write_json(Path(sys.argv[1])/'complete.json', dict(success=True, checkpoints=6,
    step=5000, diagnostic_jobs=18, finetune_jobs=54,
    completed_utc=datetime.now(timezone.utc).isoformat()))
PY
echo FINAL_ONLY_DIAGNOSTICS_AND_FINETUNING_COMPLETE
