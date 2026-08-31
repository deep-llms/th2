#1 +30+a
#th2-readonly-audit-frequency-binned-ppl-a03-and-burn-20260831-a01
set -euo pipefail

TASK_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/frequency_binned_ppl_four_10k_20260831_a03
TASK_REPORT="$TASK_ROOT/frequency_binned_ppl.json"
TASK_EXPORT=/mnt/local/_outputs/@PROJECT@/result_exports/frequency_binned_ppl_four_10k_20260831_a03.tar.gz
TASK_BURN=/tmp/llm_pretrain_burn.py
TASK_BURN_LOG=/tmp/llm_pretrain_burn_all_gpus.log
TASK_BURN_SHA=2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174

echo '=== result files ==='
date -u
test -x "$TASK_PYTHON"
test -s "$TASK_REPORT"
test -s "$TASK_ROOT/frequency_binned_ppl.md"
test -s "$TASK_ROOT/files.txt"
test -s "$TASK_EXPORT"
test -s "$TASK_EXPORT.sha256"
sha256sum -c "$TASK_EXPORT.sha256"
find "$TASK_ROOT/merged" -maxdepth 1 -type f -name '*_eval_ppl_bytoken.npz' -printf '%f %s bytes\n' | sort

"$TASK_PYTHON" - "$TASK_REPORT" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text())
expected = {
    "dense_ddp_false", "dense_ddp_default", "groupreduce_t4", "nested_ladder_t4"
}
assert report["metadata"]["languages"] == ["ar", "de", "en", "ru", "vi", "zh"]
assert report["level_sets"]["type_count"] == [2048, 6144, 24576, 119168]
assert report["mass_bins"]["type_count"] == [19130, 43898, 88908]
for section_name in ("level_sets", "mass_bins"):
    section = report[section_name]
    assert set(section["ppl"]) == expected
    assert abs(sum(section["training_mass_share"]) - 1.0) < 1e-12
    assert abs(sum(section["eval_token_share"]) - 1.0) < 1e-12
    print(section_name)
    for name, values in section["ppl"].items():
        assert all(value is not None and math.isfinite(value) for value in values)
        print(f"  {name}: " + ", ".join(f"{value:.6f}" for value in values))
    for name, gap in section["gap_vs_reference"].items():
        assert abs(sum(gap["per_bin_contribution"]) - gap["total_mean_log_ppl_gap"]) < 1e-12
print("FREQUENCY_BINNED_REPORT_READONLY_AUDIT_OK")
PY

echo '=== no evaluation process remains ==='
if ps -eo cmd= | grep -q '[p]pl_bytoken.py'; then
  echo 'ERROR: by-token evaluation process remains' >&2
  exit 1
fi

echo '=== exact burn identity and eight-GPU ownership ==='
echo "$TASK_BURN_SHA  $TASK_BURN" | sha256sum -c -
test -s "$TASK_BURN_LOG"
mapfile -t TASK_GPU_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d;s/[[:space:]]//g' | sort -nu
)
test "${#TASK_GPU_PIDS[@]}" -eq 8
TASK_LAUNCHER=''
for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
  test "$TASK_PID" != 1
  TASK_PPID="$(awk '/^PPid:/ {print $2}' "/proc/$TASK_PID/status")"
  test "$TASK_PPID" != 1
  if [[ -z "$TASK_LAUNCHER" ]]; then
    TASK_LAUNCHER="$TASK_PPID"
  else
    test "$TASK_PPID" = "$TASK_LAUNCHER"
  fi
done
TASK_LAUNCHER_CMD="$(tr '\0' ' ' < "/proc/$TASK_LAUNCHER/cmdline")"
[[ "$TASK_LAUNCHER_CMD" == *"$TASK_BURN"* ]]

for TASK_MARKER in \
  gpu_burn_ready world_size=8 collective_probe_sum=36 \
  comm_total_mib=1137 comm_bucket_mib=25 approx_step_seconds=0.750; do
  TASK_COUNT="$(grep -oF "$TASK_MARKER" "$TASK_BURN_LOG" | wc -l || true)"
  test "$TASK_COUNT" -eq 8
done
grep -Fq 'gpu_burn_progress' "$TASK_BURN_LOG"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader
echo "burn_launcher=$TASK_LAUNCHER gpu_workers=${TASK_GPU_PIDS[*]}"
echo 'FREQUENCY_BINNED_PPL_AND_CORRECT_ALL_EIGHT_GPU_BURNS_AUDIT_OK'
