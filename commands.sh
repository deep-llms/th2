#1 +30+a
#th2-ohmv-warm-downstream-path-inspect-20261118-a01
set -euo pipefail
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PYCODE'
import json,pathlib
root=pathlib.Path('/mnt/local/_data/deep-llms_th2/warm_downstream_20261118')
print('PATH_INSPECT',json.dumps({'exists':root.exists(),'entries':[{'path':str(p),'dir':p.is_dir(),'bytes':p.stat().st_size if p.is_file() else None} for p in root.rglob('*')] if root.exists() else [],'root_dirs':[str(p) for p in root.parent.iterdir()]}))
PYCODE
