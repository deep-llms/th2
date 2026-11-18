#1 +60+a
#th2-ohmv-warm-downstream-probe-trace-20261118-a01
set -euo pipefail
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PYCODE'
import pathlib,json
b=pathlib.Path('/mnt/local/_outputs/deep-llms_th2/warm_downstream_20261118')
print('PROBE_TRACE',json.dumps({p:(b/p).read_text()[-16000:] if (b/p).exists() else None for p in ['probe-numerical.log','failure.json','terminal.json','requests.json.log']}))
PYCODE
