#1 +30+a
#th2-pq5r-dot-l47-curves-20261128-a01
set -uo pipefail
printf 'READ_ONLY_L47_CURVES (never signals a process)\n'
date -u
base=/mnt/local/_outputs/deep-llms_th2/dot_l47_20261128/run
for d in "$base"/s*/*_4x; do
  python3 - "$d" <<'PYC'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
f = d/'calibration.jsonl'
if not f.is_file():
    print(d.parent.name + '/' + d.name, 'no calibration'); sys.exit()
rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
pts = [(r['step'], r['nll']) for r in rows]
keep = [p for p in pts if p[0] % 4096 == 0 or p == pts[-1]]
print(d.parent.name + '/' + d.name, ' '.join(f"{s}:{v:.3f}" for s, v in keep))
PYC
done
grep -h -i -E "nan|inf|spike|overflow" "$base"/logs/fit-*d1536*.log 2>/dev/null | head -5 || true
