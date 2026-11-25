#1 +240+a
#th2-ohmv-dot-shared-heads-f03-20261125
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_shared_heads_20261125_f03" ]; then echo F03_ALREADY_LAUNCHED; cat $out/dot_shared_heads_20261125_f03/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_compact_t_20261125_f02')
t = json.loads((base/'terminal.json').read_text())
print('F02_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
# f02's launch log was never pulled: check its frozen source carries exactly the f02 package pins
src = json.loads((base/'execution/source.json').read_text())
pins = {"efficient_llm/warm_order/dot_stop_sandwich.py": "e922c09b722645eba6e92bcddf8fb8d8a4c7264e74a878114bf1df61cf77fd24",
        "efficient_llm/warm_order/loop_screen.py": "1b047d9e7b43ca164578d8e51b49a3840e594675b9137002bfee411636c85129",
        "scripts/loop_screen_study.py": "daa51f02738d52342bf26b84901053e857b75ea3d37039f82d4e122b78cce4af",
        "scripts/dot_sandwich_study.py": "30d2cff3f7420e5812d962d790d14748a46e80f2c6bbcba943a0fd376563dcc5"}
assert all(src.get(k) == v for k, v in pins.items()), 'f02 source is not the f02 package'
PY
receipt=$(sha256sum $out/dot_compact_t_20261125_f02/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
[ "$receipt" = 30b014a5bf75b6712d7810bd7013f4a242582dfd5321be126c938970f01e73b7 ]
prior=$(sha256sum $out/dot_compact_t_20261125_f02/execution/source.json | cut -d' ' -f1)
echo "F02_SOURCE $prior"
pkg="$PWD/resources/dot_f03_20261125.tar.xz"
echo "d9ca456a47a6b85ceb055c8e78a61bd20f0b6bb83b97becac6962c18959b6f29  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f03_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_compact_t_20261125_f02/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_shared_heads_20261125_f03 \
  --old $out/dot_compact_t_20261125_f02/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_baselines.py efficient_llm/warm_order/shared_heads.py efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_baselines.py efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_compact_t_20261125_f02/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_shared_heads_20261125_f03')
jobs = [(0, 'dense4_shrep'), (1, 'dense4_shrep'), (0, 'loop_cycle_d3_shrep'), (1, 'loop_cycle_d3_shrep'),
        (0, 'loop_cycle_d3_shcat'), (0, 'loop_cycle_d3_shvadd'), (0, 'loop_cycle_d3_shoadd'), (0, 'swst_fixed_d3_tmlp')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F03_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F03_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F03_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
