#1 +60+a
#th2-verify-culturax-download-files-20260902-a17
set -euo pipefail

TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_RAW_DIR=/mnt/local/_data/@PROJECT@/data/raw
TASK_MANIFEST="$PWD/resources/culturax_raw_manifest.tsv"

echo '=== verify downloaded CulturaX mirror against exact H100 manifest ==='
date -u
hostname
test -x "$TASK_PYTHON"
test -d "$TASK_RAW_DIR"
test -s "$TASK_MANIFEST"

"$TASK_PYTHON" - "$TASK_RAW_DIR" "$TASK_MANIFEST" <<'PY'
import collections
import sys
from pathlib import Path

raw = Path(sys.argv[1])
manifest = Path(sys.argv[2])
expected = {}
for line in manifest.read_text().splitlines():
    if not line or line.startswith("#"):
        continue
    digest, size, relative = line.split("\t")
    assert len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
    assert relative not in expected
    expected[relative] = int(size)

actual_paths = {
    path.relative_to(raw).as_posix(): path
    for path in raw.rglob("*.parquet")
    if path.is_file()
}
assert set(actual_paths) == set(expected), {
    "missing": sorted(set(expected) - set(actual_paths)),
    "unexpected": sorted(set(actual_paths) - set(expected)),
}
for relative, wanted_size in expected.items():
    actual_size = actual_paths[relative].stat().st_size
    assert actual_size == wanted_size, (relative, actual_size, wanted_size)

counts = collections.Counter(relative.split("/", 1)[0] for relative in expected)
assert counts == {"en": 50, "vi": 5, "zh": 5, "ru": 5, "de": 5, "ar": 5}, counts
total = sum(expected.values())
assert len(expected) == 75
assert total == 166_107_112_571, total
print(f"files={len(expected)} bytes={total} per_language={dict(sorted(counts.items()))}")
print("CULTURAX_PATHS_AND_SIZES_OK")
PY

echo '=== verify SHA-256 of all 166 GB / 75 parquet files ==='
(
    cd "$TASK_RAW_DIR"
    awk '!/^#/ {print $1 "  " $3}' "$TASK_MANIFEST" | sha256sum --check --strict --quiet -
)
echo 'CULTURAX_ALL_SHA256_OK'

echo '=== GPU burns remain active ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits
echo 'TH2 CULTURAX DOWNLOAD FILE VERIFICATION OK'
