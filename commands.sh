#1 +240+a
#th2-verify-offline-eval-finetune-benchmarks-20260904-a01
set -euo pipefail

TASK_PROJECT_DIR=/mnt/local/@PROJECT@
TASK_CONDA=/mnt/local/conda-py311/bin/conda
TASK_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_BENCH_ROOT=/mnt/local/_data/@PROJECT@/benchmarks/hf

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LM_EVAL_DATASET_ROOT="$TASK_BENCH_ROOT"

cd "$TASK_PROJECT_DIR"
date -u
hostname
test -x "$TASK_CONDA"
test -x "$TASK_EVAL_PYTHON"
eval "$("$TASK_CONDA" shell.bash hook)"
conda activate eval
test "${CONDA_DEFAULT_ENV:-}" = eval
test "$(command -v python3.11)" = "$TASK_EVAL_PYTHON"

echo '=== nonempty downloaded snapshots ==='
TASK_DATASET_DIRS=(
    facebook/xnli
    facebook/belebele
    cambridgeltl/xcopa
    juletxara/xstory_cloze
    google-research-datasets/paws-x
    Rowan/hellaswag
    alexandrainst/m_hellaswag
    allenai/ai2_arc
    alexandrainst/m_arc
)
for TASK_RELPATH in "${TASK_DATASET_DIRS[@]}"; do
    TASK_DATASET_DIR="$TASK_BENCH_ROOT/$TASK_RELPATH"
    test -d "$TASK_DATASET_DIR"
    TASK_FILE_COUNT="$(find "$TASK_DATASET_DIR" -type f | wc -l)"
    TASK_BYTE_COUNT="$(find "$TASK_DATASET_DIR" -type f -printf '%s\n' \
        | awk '{sum += $1} END {print sum + 0}')"
    test "$TASK_FILE_COUNT" -gt 0
    test "$TASK_BYTE_COUNT" -gt 0
    printf 'dataset=%-45s files=%-5s bytes=%s\n' \
        "$TASK_RELPATH" "$TASK_FILE_COUNT" "$TASK_BYTE_COUNT"
done

echo '=== load every configured eval split and finetune train split offline ==='
"$TASK_EVAL_PYTHON" - <<'PY'
import os
from pathlib import Path
from datasets import load_dataset
from eval.benchmarks import TASK_CONFIGS, patch_lm_eval_dataset_paths

root = Path(os.environ['LM_EVAL_DATASET_ROOT'])
cases = [
    *[('facebook/xnli', lang, 'validation', f'xnli_{lang}')
      for lang in ('en', 'vi', 'zh', 'ru', 'de', 'ar')],
    *[('facebook/belebele', lang, 'test', f'belebele_{lang}')
      for lang in ('eng_Latn', 'vie_Latn', 'zho_Hans', 'rus_Cyrl', 'deu_Latn', 'arb_Arab')],
    *[('cambridgeltl/xcopa', lang, 'validation', f'xcopa_{lang}')
      for lang in ('vi', 'zh')],
    *[('juletxara/xstory_cloze', lang, 'eval', f'xstorycloze_{lang}')
      for lang in ('en', 'ar', 'ru', 'zh')],
    *[('google-research-datasets/paws-x', lang, 'validation', f'paws_{lang}')
      for lang in ('en', 'de', 'zh')],
    ('Rowan/hellaswag', None, 'validation', 'hellaswag'),
    *[('alexandrainst/m_hellaswag', lang, 'val', f'hellaswag_{lang}')
      for lang in ('ar', 'de', 'ru', 'vi')],
]
assert len(cases) == 26
for relpath, config, split, task in cases:
    dataset = load_dataset(str(root / relpath), name=config, split=split, streaming=True)
    row = next(iter(dataset))
    assert isinstance(row, dict) and row, (task, row)
    print(f'EVAL_DATA_OK task={task} split={split}')

training_cases = [
    ('Rowan/hellaswag', None, 'train', 'hellaswag'),
    ('allenai/ai2_arc', 'ARC-Easy', 'train', 'arc_easy'),
    ('facebook/xnli', 'en', 'train', 'xnli'),
]
for relpath, config, split, task in training_cases:
    dataset = load_dataset(str(root / relpath), name=config, split=split, streaming=True)
    row = next(iter(dataset))
    assert isinstance(row, dict) and row, (task, row)
    print(f'FINETUNE_TRAIN_DATA_OK task={task} split={split}')

patch_lm_eval_dataset_paths(str(root))
tasks = [task for group in TASK_CONFIGS.values() for task in group]
assert len(tasks) == len(set(tasks)) == 26, tasks
print('OFFLINE_INPUTS_OK eval_tasks=26 finetune_training_datasets=3')
PY

echo '=== verify current burns were not disturbed ==='
TASK_ALL_PIDS=()
for TASK_GPU_INDEX in 0 1 2 3 4 5 6 7; do
    mapfile -t TASK_ONE_PIDS < <(
        nvidia-smi -i "$TASK_GPU_INDEX" --query-compute-apps=pid \
            --format=csv,noheader,nounits \
            | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;/^$/d' | sort -nu
    )
    test "${#TASK_ONE_PIDS[@]}" -eq 1
    TASK_PID="${TASK_ONE_PIDS[0]}"
    test "$TASK_PID" -ne 1
    TASK_PPID="$(awk '/^PPid:/ {print $2}' "/proc/$TASK_PID/status")"
    TASK_PARENT="$(tr '\0' ' ' < "/proc/$TASK_PPID/cmdline")"
    [[ "$TASK_PARENT" == *'/tmp/llm_pretrain_burn.py'* ]]
    TASK_ALL_PIDS+=("$TASK_PID")
done
mapfile -t TASK_UNIQUE_PIDS < <(printf '%s\n' "${TASK_ALL_PIDS[@]}" | sort -nu)
test "${#TASK_UNIQUE_PIDS[@]}" -eq 8
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
echo 'TH2 OFFLINE EVAL AND FINETUNE BENCHMARKS VERIFIED; BURNS UNTOUCHED'
