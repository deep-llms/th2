#1 +45+a
#th2-pq5r-dot-l34-health-or-wrapup-20261128-a03
set -uo pipefail
printf 'READ_ONLY_L34_HEALTH_OR_WRAPUP (never signals a process)\n'
date -u
out=/mnt/local/_outputs/deep-llms_th2/dot_l34_20261128
base=$out/run
PY=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
cat "$base/jobs.json" 2>&1 | head -40
for d in "$base"/s*/*_4x; do
  printf '%s ' "${d#$base/}"
  python3 -c "import json,sys;p=json.load(open(sys.argv[1]));print('step',p.get('step'),'/',p.get('total'))" "$d/progress.json" 2>/dev/null || printf 'no progress\n'
  test -f "$d/summary.json" && python3 -c "import json,sys;v=json.load(open(sys.argv[1]))['nll'];print('  SUMMARY', ' '.join(f'{k}:{v[k]:.4f}' for k in ('native','r1','r2','r4','r8','r16','r32') if k in v))" "$d/summary.json"
done
grep -l -E "Traceback|Error" "$base"/logs/*.log 2>/dev/null || printf 'no errors in job logs\n'
tail -n 4 "$out/supervisor.log" | cut -c1-200
for f in failure.json handoff-failure.json; do test -e "$base/$f" && { printf '%s: ' "$f"; cat "$base/$f"; }; done
if test -f "$base/terminal.json"; then
  printf 'TERMINAL '; cat "$base/terminal.json" | head -c 600; printf '\n'
  if python3 -c "import json,sys;t=json.load(open(sys.argv[1]));sys.exit(0 if t.get('native_restored') else 1)" "$base/terminal.json"; then
    test -e "$out/inspection-a02.json" || "$PY" scripts/node_gpu_inspect.py "$out/inspection-a02.json"
    sha256sum "$out/inspection-a02.json"
    test -e "$out/l34_results.tar.gz" || "$PY" scripts/node_bundle_results.py --base "$base" --out "$out/l34_results"
    ls -la "$out"/l34_results.* && sha256sum "$out/l34_results.tar.gz"
  else
    printf 'NATIVE_NOT_RESTORED: no inspection, no next batch\n'
  fi
fi
tmux ls 2>&1
ls -la /mnt/local/_gpu_guard/ | tail -n +2
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
