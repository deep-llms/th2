#1 +120+a
#th2-verify-offline-lm-eval-benchmarks-20260824
#!/usr/bin/env bash
set -euo pipefail

BENCH_ROOT=/mnt/local/_data/@PROJECT@/benchmarks/hf
EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo '=== benchmark snapshot inventory ==='
date -u
hostname
test -x "$EVAL_PYTHON"

DATASET_DIRS=(
    facebook/xnli
    facebook/belebele
    cambridgeltl/xcopa
    juletxara/xstory_cloze
    google-research-datasets/paws-x
    Rowan/hellaswag
    alexandrainst/m_hellaswag
)

for relpath in "${DATASET_DIRS[@]}"; do
    dataset_dir="$BENCH_ROOT/$relpath"
    test -d "$dataset_dir"
    file_count=$(find "$dataset_dir" -type f | wc -l)
    byte_count=$(find "$dataset_dir" -type f -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')
    test "$file_count" -gt 0
    test "$byte_count" -gt 0
    tree_hash=$(find "$dataset_dir" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
    printf '%-45s files=%-5s bytes=%-12s sha256_tree=%s\n' "$relpath" "$file_count" "$byte_count" "$tree_hash"
done

echo '=== offline datasets loading for every configured benchmark ==='
"$EVAL_PYTHON" - <<'PY'
from pathlib import Path

from datasets import load_dataset

root = Path('/mnt/local/_data/@PROJECT@/benchmarks/hf')
cases = [
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

assert len(cases) == 26
for relpath, config, split, task in cases:
    path = root / relpath
    dataset = load_dataset(str(path), name=config, split=split, streaming=True)
    row = next(iter(dataset))
    assert isinstance(row, dict) and row, (task, row)
    print(f"OK task={task:<24} config={str(config):<10} split={split:<10} columns={sorted(row)}")

print('OFFLINE BENCHMARK VERIFICATION OK: 7 repositories, 26 tasks')
PY

echo '=== confirm GPU burns were not touched ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
echo 'TH2 OFFLINE LM-EVAL BENCHMARKS VERIFIED'
