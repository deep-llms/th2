#1 +180+a
#th2-clean-cancelled-four-model-outputs-and-verify-benchmarks-20260830-a01
set -euo pipefail

TASK_PROJECT_DIR=/mnt/local/@PROJECT@
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_BENCH_ROOT=/mnt/local/_data/@PROJECT@/benchmarks/hf
TASK_CONDA=/mnt/local/conda-py311/bin/conda
TASK_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_nested_groupreduce_two_dense_10k_20260830"
TASK_EVAL_LAUNCH_LOG="$TASK_OUTPUT_BASE/eval_parallel_nested_groupreduce_two_dense_10k_20260830.log"
TASK_CHECKPOINTS=(
    "$TASK_OUTPUT_BASE/nested_ladder_tied_t4/checkpoint-10000"
    "$TASK_OUTPUT_BASE/groupreduce_matched_nested_tied_t4/checkpoint-10000"
    "$TASK_OUTPUT_BASE/dense_tied_baseline_b200/checkpoint-10000"
    "$TASK_OUTPUT_BASE/dense_tied_baseline_b200_ddp_default/checkpoint-10000"
)

die() {
    echo "ERROR: $*" >&2
    exit 1
}

echo '=== preflight: cancelled workflow must still be stopped ==='
date -u
hostname
cd "$TASK_PROJECT_DIR"
test -x "$TASK_CONDA"
test -x "$TASK_EVAL_PYTHON"
test -d "$TASK_OUTPUT_BASE"
test -d "$TASK_BENCH_ROOT"

"$TASK_EVAL_PYTHON" - "$TASK_FINETUNE_OUTPUT" "${TASK_CHECKPOINTS[@]}" <<'PY'
import os
import sys

finetune_output, *checkpoints = sys.argv[1:]


def read_args(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [part.decode(errors="replace") for part in raw.rstrip(b"\0").split(b"\0") if part]


def has_script(args, suffix):
    return any(arg == suffix or arg.endswith("/" + suffix) for arg in args[1:])


matches = []
skip = {1, os.getpid(), os.getppid()}
for entry in os.scandir("/proc"):
    if not entry.name.isdigit():
        continue
    pid = int(entry.name)
    if pid in skip:
        continue
    args = read_args(pid)
    if not args:
        continue
    matched = (
        (has_script(args, "eval/eval_parallel.py") and all(path in args for path in checkpoints))
        or (has_script(args, "eval/eval_checkpoint.py") and any(path in args for path in checkpoints))
        or (has_script(args, "finetune/run_all.py") and finetune_output in args)
        or (
            has_script(args, "finetune/train.py")
            and finetune_output in args
            and any(path in args for path in checkpoints)
        )
    )
    if matched:
        matches.append((pid, args))

if matches:
    for pid, args in matches:
        print(f"ACTIVE_TARGET pid={pid} argv={args}", file=sys.stderr)
    raise SystemExit("refusing cleanup while the cancelled workflow still has live workers")
print("NO_CANCELLED_EVAL_OR_FINETUNE_PROCESSES")
PY

echo '=== inventory only the cancelled workflow outputs before removal ==='
TASK_REMOVAL_FILES=("$TASK_EVAL_LAUNCH_LOG")
for TASK_CHECKPOINT in "${TASK_CHECKPOINTS[@]}"; do
    test -d "$TASK_CHECKPOINT" || die "checkpoint missing: $TASK_CHECKPOINT"
    TASK_REMOVAL_FILES+=(
        "$TASK_CHECKPOINT/eval.log"
        "$TASK_CHECKPOINT/eval_ppl.json"
        "$TASK_CHECKPOINT/eval_benchmarks.json"
    )
done
for TASK_PATH in "${TASK_REMOVAL_FILES[@]}"; do
    case "$TASK_PATH" in
        "$TASK_OUTPUT_BASE"/*) ;;
        *) die "refusing output path outside project output root: $TASK_PATH" ;;
    esac
    if [[ -e "$TASK_PATH" || -L "$TASK_PATH" ]]; then
        test -f "$TASK_PATH" || die "expected a regular output file: $TASK_PATH"
        stat --printf='REMOVE_FILE %n %s bytes\n' "$TASK_PATH"
    else
        echo "ABSENT_FILE $TASK_PATH"
    fi
done
case "$TASK_FINETUNE_OUTPUT" in
    "$TASK_OUTPUT_BASE"/finetune_nested_groupreduce_two_dense_10k_20260830) ;;
    *) die "unexpected finetune removal target: $TASK_FINETUNE_OUTPUT" ;;
esac
if [[ -e "$TASK_FINETUNE_OUTPUT" || -L "$TASK_FINETUNE_OUTPUT" ]]; then
    test -d "$TASK_FINETUNE_OUTPUT" || die "expected finetune directory: $TASK_FINETUNE_OUTPUT"
    echo "REMOVE_DIRECTORY $TASK_FINETUNE_OUTPUT"
    find "$TASK_FINETUNE_OUTPUT" -type f -printf '%s\n' \
        | awk '{bytes += $1; files += 1} END {printf "  files=%d bytes=%.0f\n", files, bytes}'
else
    echo "ABSENT_DIRECTORY $TASK_FINETUNE_OUTPUT"
fi

echo '=== remove only those exact cancelled-workflow outputs ==='
rm -f -- "${TASK_REMOVAL_FILES[@]}"
rm -rf -- "$TASK_FINETUNE_OUTPUT"

echo '=== verify cancelled-workflow outputs are gone and checkpoints remain ==='
for TASK_PATH in "${TASK_REMOVAL_FILES[@]}"; do
    [[ ! -e "$TASK_PATH" && ! -L "$TASK_PATH" ]] \
        || die "output remains after cleanup: $TASK_PATH"
done
[[ ! -e "$TASK_FINETUNE_OUTPUT" && ! -L "$TASK_FINETUNE_OUTPUT" ]] \
    || die "finetune output remains after cleanup: $TASK_FINETUNE_OUTPUT"
for TASK_CHECKPOINT in "${TASK_CHECKPOINTS[@]}"; do
    test -s "$TASK_CHECKPOINT/config.json"
    test -s "$TASK_CHECKPOINT/model.safetensors"
    test -s "$TASK_CHECKPOINT/trainer_state.json"
done
echo 'CANCELLED_WORKFLOW_OUTPUTS_CLEAN'

echo '=== activate the offline evaluation environment ==='
eval "$("$TASK_CONDA" shell.bash hook)"
conda activate eval
test "${CONDA_DEFAULT_ENV:-}" = eval
test "$(command -v python3.11)" = "$TASK_EVAL_PYTHON"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LM_EVAL_DATASET_ROOT="$TASK_BENCH_ROOT"
export TOKENIZERS_PARALLELISM=false

TASK_VERIFY_CACHE=$(mktemp -d /tmp/th2-benchmark-integrity-verify.XXXXXX)
cleanup_verify_cache() {
    rm -rf -- "$TASK_VERIFY_CACHE"
}
trap cleanup_verify_cache EXIT

echo '=== verify repository files, hashes, and absence of unresolved LFS pointers ==='
"$TASK_EVAL_PYTHON" - "$TASK_BENCH_ROOT" <<'PY'
import hashlib
import os
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
repositories = (
    "facebook/xnli",
    "facebook/belebele",
    "cambridgeltl/xcopa",
    "juletxara/xstory_cloze",
    "google-research-datasets/paws-x",
    "Rowan/hellaswag",
    "alexandrainst/m_hellaswag",
    "allenai/ai2_arc",
    "alexandrainst/m_arc",
)

for relative in repositories:
    repository = (root / relative).resolve()
    assert repository.is_dir(), repository
    assert repository == root / relative, (repository, root / relative)
    files = sorted(path for path in repository.rglob("*") if path.is_file())
    assert files, repository
    total_bytes = 0
    digest = hashlib.sha256()
    for path in files:
        resolved = path.resolve()
        assert resolved == path or str(resolved).startswith(str(repository) + os.sep), path
        relative_file = path.relative_to(repository).as_posix()
        size = path.stat().st_size
        total_bytes += size
        digest.update(relative_file.encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            prefix = handle.read(200)
            assert not prefix.startswith(b"version https://git-lfs.github.com/spec/v1"), (
                "unresolved Git LFS pointer",
                path,
            )
            digest.update(prefix)
            while chunk := handle.read(4 * 1024 * 1024):
                digest.update(chunk)
    assert total_bytes > 0, repository
    print(
        f"REPOSITORY_FILES_OK dataset={relative} files={len(files)} "
        f"bytes={total_bytes} aggregate_sha256={digest.hexdigest()}"
    )
print(f"NINE_DATASET_REPOSITORIES_OK count={len(repositories)}")
PY

echo '=== load every project-required benchmark split from a fresh cache ==='
HF_DATASETS_CACHE="$TASK_VERIFY_CACHE" "$TASK_EVAL_PYTHON" - <<'PY'
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
import lm_eval
import yaml

from eval.benchmarks import TASK_CONFIGS, patch_lm_eval_dataset_paths

root = Path(os.environ["LM_EVAL_DATASET_ROOT"])
cache_dir = os.environ["HF_DATASETS_CACHE"]


def validate_split(repository, config, split, label):
    dataset = load_dataset(
        str(root / repository),
        name=config,
        split=split,
        cache_dir=cache_dir,
    )
    assert isinstance(dataset, Dataset), (label, type(dataset))
    assert len(dataset) > 0, label
    first = dataset[0]
    last = dataset[len(dataset) - 1]
    assert isinstance(first, dict) and first, (label, first)
    assert isinstance(last, dict) and last, (label, last)
    print(
        f"DATA_SPLIT_OK label={label} repository={repository} "
        f"config={config or '<default>'} split={split} rows={len(dataset)}"
    )


evaluation_cases = [
    *[("facebook/xnli", lang, "validation", f"xnli_{lang}")
      for lang in ("en", "vi", "zh", "ru", "de", "ar")],
    *[("facebook/belebele", lang, "test", f"belebele_{lang}")
      for lang in ("eng_Latn", "vie_Latn", "zho_Hans", "rus_Cyrl", "deu_Latn", "arb_Arab")],
    *[("cambridgeltl/xcopa", lang, "validation", f"xcopa_{lang}")
      for lang in ("vi", "zh")],
    *[("juletxara/xstory_cloze", lang, "eval", f"xstorycloze_{lang}")
      for lang in ("en", "ar", "ru", "zh")],
    *[("google-research-datasets/paws-x", lang, "validation", f"paws_{lang}")
      for lang in ("en", "de", "zh")],
    ("Rowan/hellaswag", None, "validation", "hellaswag"),
    *[("alexandrainst/m_hellaswag", lang, "val", f"hellaswag_{lang}")
      for lang in ("ar", "de", "ru", "vi")],
]
assert len(evaluation_cases) == 26
for case in evaluation_cases:
    validate_split(*case)

training_cases = (
    ("Rowan/hellaswag", None, "train", "finetune_train_hellaswag"),
    ("allenai/ai2_arc", "ARC-Easy", "train", "finetune_train_arc_easy"),
    ("facebook/xnli", "en", "train", "finetune_train_xnli"),
)
for case in training_cases:
    validate_split(*case)

patch_lm_eval_dataset_paths(str(root))
tasks = [task for group in TASK_CONFIGS.values() for task in group]
assert len(tasks) == len(set(tasks)) == 26, tasks

# ARC-Easy and its five multilingual counterparts are used after each ARC
# finetuning job. Read dataset_name from this installed lm-eval version, then
# fully materialize every available split from the local repositories.
task_root = Path(lm_eval.__file__).parent / "tasks"
arc_specs = [
    ("arc/arc_easy.yaml", "allenai/ai2_arc", "arc_easy"),
    *[(f"okapi/arc_multilingual/arc_{lang}.yaml", "alexandrainst/m_arc", f"arc_{lang}")
      for lang in ("ar", "de", "ru", "vi", "zh")],
]
for yaml_relative, repository, label in arc_specs:
    yaml_path = task_root / yaml_relative
    assert yaml_path.is_file(), yaml_path
    with yaml_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    dataset_name = config.get("dataset_name")
    assert dataset_name, (yaml_path, config)
    datasets = load_dataset(
        str(root / repository),
        name=dataset_name,
        cache_dir=cache_dir,
    )
    assert isinstance(datasets, DatasetDict), (label, type(datasets))
    assert datasets, label
    for split, dataset in datasets.items():
        assert len(dataset) > 0, (label, split)
        assert isinstance(dataset[0], dict) and dataset[0], (label, split)
        assert isinstance(dataset[len(dataset) - 1], dict) and dataset[len(dataset) - 1], (
            label,
            split,
        )
        print(
            f"ARC_EVAL_DATA_OK label={label} repository={repository} "
            f"config={dataset_name} split={split} rows={len(dataset)}"
        )

print(
    "ALL_REQUIRED_BENCHMARK_DATA_VERIFIED "
    "repositories=9 zero_shot_tasks=26 finetune_train_datasets=3 arc_eval_tasks=6"
)
PY

echo '=== final GPU state (read-only) ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits || true
echo 'TH2 CANCELLED EVAL/FINETUNE OUTPUTS REMOVED; ALL BENCHMARK DATA VERIFIED'
