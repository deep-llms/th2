#1 +20+a
#th2-pq5r-dot-l34-health-20261128-a01
set -uo pipefail
printf 'READ_ONLY_L34_HEALTH (does not touch the running supervisor or its jobs)\n'
date -u
base=/mnt/local/_outputs/deep-llms_th2/dot_l34_20261128/run
cat "$base/jobs.json" 2>&1 | head -30
for f in failure.json handoff-failure.json terminal.json; do test -e "$base/$f" && { printf '%s: ' "$f"; cat "$base/$f"; }; done
for d in "$base"/s*/*_4x; do
  printf '%s ' "${d#$base/}"
  python3 -c "import json,sys;p=json.load(open(sys.argv[1]));print('step',p.get('step'),'/',p.get('total'))" "$d/progress.json" 2>/dev/null || printf 'no progress\n'
  tail -n 1 "$d/calibration.jsonl" 2>/dev/null | cut -c1-120
done
grep -l -E "Traceback|Error" "$base"/logs/*.log 2>/dev/null || printf 'no errors in job logs\n'
tail -n 3 /mnt/local/_outputs/deep-llms_th2/dot_l34_20261128/supervisor.log | cut -c1-200
ls -la /mnt/local/_gpu_guard/ | tail -n +2
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader
