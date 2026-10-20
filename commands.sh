#1 +30+a
#th2-6y3o-fa2-local-linker-name-20261020-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train_fa2/bin/python3.11 - <<'CHECK'
from pathlib import Path
import sys,json,hashlib,datetime
prefix=Path('/mnt/local/conda-py311/envs/train_fa2')
assert Path(sys.prefix).resolve()==prefix
lib=prefix/'lib/python3.11/site-packages/nvidia/cu13/lib'
assert lib.resolve().is_relative_to(prefix)
target=lib/'libcudart.so.13'
link=lib/'libcudart.so'
assert target.is_file() and target.resolve().is_relative_to(prefix)
created=False
if link.exists() or link.is_symlink():
    assert link.resolve()==target.resolve(), 'Unexpected existing linker name; refuse overwrite'
else:
    link.symlink_to(target.name)
    created=True
assert link.resolve()==target.resolve()
with target.open('rb') as stream:
    sha=hashlib.file_digest(stream,'sha256').hexdigest()
report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),created=created,link=str(link),target=str(target),target_sha256=sha)
work=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-fa2-build-20261019-a02')
assert work.is_dir()
note=work/'linker_name.json'
with note.open('x') as stream:
    json.dump(report,stream,indent=2)
print('FA2_LOCAL_LINKER_NAME '+json.dumps(report),flush=True)
print('FA2_OBJECT_FILES',sum(1 for _ in work.glob('source/**/build/**/*.o')),flush=True)
CHECK
