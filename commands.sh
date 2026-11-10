#1 +20+a
#th2-wc69-family-f9-source-attestation-20261110-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import json
from pathlib import Path
from datetime import datetime,timezone
from efficient_llm.diagnostic_data import sha256
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-20261110-a01')
expected=json.loads(Path('resources/a2_family_f9_source_attestation_20261110.json').read_text())
manifest=root/'source-manifest.json'
assert sha256(manifest)==expected['original_manifest_sha256']
m=json.loads(manifest.read_text());assert m['code_commit']==''
checks={name:sha256(root/'source'/name)==h==expected['files'].get(name) for name,h in m['files'].items()}
assert len(checks)==253 and all(checks.values()),[p for p,v in checks.items() if not v]
out=root/'results/a2_family_20261107/downstream/SOURCE_COMMIT_ATTESTATION.json'
value=dict(spec_sha256=expected['spec_sha256'],utc=datetime.now(timezone.utc).isoformat(),verified=True,actual_source_commit=expected['commit'],checks=checks,source_manifest_sha256=sha256(manifest),original_code_commit='',reason='Node synced source has no .git; launch command substitution left commit blank. Every frozen source file independently matches the deployed Git commit. Original manifest and running processes unchanged.',launch_receipt_sha256=expected['launch_receipt_sha256'],health_receipt_sha256=expected['health_receipt_sha256'])
with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print('F9_SOURCE_ATTESTED',expected['commit'],len(checks),sha256(out),flush=True)
PY
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.a2_family_f9_health --root /mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-20261110-a01
