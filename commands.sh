#1 +30+a
#th2-swt-verify-package-all-eval-20260910-1759
set -euo pipefail
date -u
hostname
cd /mnt/local/deep-llms_th2
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
CUDA_VISIBLE_DEVICES='' /mnt/local/conda-py311/envs/swt_eval/bin/python -B - <<'PY'
import hashlib, json, zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
root = Path('/mnt/local/_outputs/deep-llms_th2/swt/next_capacity_eval98_diag294_ft126_20260910_a02')
dest = Path('/mnt/local/_outputs/deep-llms_th2/swt/exports/eval98_all_20260910_a02')
lines = (root/'pipeline.log').read_text(errors='replace').splitlines()
print('PIPELINE_TAIL', '\n'.join(lines[-15:]))
for rel in ('eval/failed.json','eval_verified.json','finetune/failed.json','finetune_verified.json','complete.json'):
    p = root/rel
    print('STAGE_MARKER', rel, p.read_text()[:1500] if p.exists() else 'absent')
if not (root/'eval_verified.json').exists():
    print('EVAL_NOT_YET_VERIFIED_NO_EXPORT')
    raise SystemExit(0)
assert json.loads((root/'eval_verified.json').read_text())['success'] is True
assert not dest.exists()
from scripts.next_capacity_eval_handoff import verify_eval
dest.mkdir(parents=True, exist_ok=False)
verify_eval(SimpleNamespace(reference=root/'inputs_verified.json', output=root/'eval',
                           summary=dest/'export_verification.json'))
complete = json.loads((root/'eval/complete.json').read_text())
assert complete['success'] is True and len(complete['completed']) == 490
paths = [Path(r['result']) for r in complete['completed']]
assert len(set(paths)) == 490
paths += [root/p for p in ('inputs_verified.json','eval_plan.json','finetune_plan.json',
                           'eval/plan.json','eval/complete.json','eval_verified.json')]
payload = {}
for p in paths:
    assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
    assert p.suffix == '.json'
    payload[p.relative_to(root).as_posix()] = p.read_bytes()
assert len(payload) == 496
archive = dest/'eval_results.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as handle:
    for name,data in sorted(payload.items()):
        handle.writestr(name,data)
assert archive.stat().st_size < 25_000_000, 'Archive exceeds runner export limit'
manifest = dict(success=True, source_run=str(root), checkpoints=98, result_files=490,
    exported_utc=datetime.now(timezone.utc).isoformat(),
    files=[dict(path=n,bytes=len(d),sha256=hashlib.sha256(d).hexdigest()) for n,d in sorted(payload.items())],
    archive=dict(name=archive.name,bytes=archive.stat().st_size,
                 sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
with (dest/'export_manifest.json').open('x') as handle:
    json.dump(manifest,handle,indent=2,allow_nan=False)
    handle.write('\n')
print('VERIFIED_EXPORT', json.dumps({k:v for k,v in manifest.items() if k!='files'}))
print('MANIFEST_SHA256',hashlib.sha256((dest/'export_manifest.json').read_bytes()).hexdigest())
print('ALL_EVAL_RESULTS_EXPORT_READY')
PY
date -u
