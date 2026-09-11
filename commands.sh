#1 +60+a
#th2-8mgy-verify-culturax-download-20260911-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-8mgy-worker-0
/mnt/local/conda-py311/envs/train_env/bin/python -B - <<'PY'
import collections, hashlib, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
root=Path('/mnt/local/_data/deep-llms_th2/data/raw')
manifest=Path('resources/culturax_raw_manifest.tsv').read_text()
expected={}
for line in manifest.splitlines():
    if not line or line.startswith('#'): continue
    digest,size,relative=line.split()
    expected[relative]=(digest,int(size))
assert len(expected)==75
actual={p.relative_to(root).as_posix() for p in root.rglob('*.parquet')}
assert actual==set(expected), dict(missing=sorted(set(expected)-actual),extra=sorted(actual-set(expected)))
assert sum(size for _,size in expected.values())==166107112571
for name,(_,size) in expected.items():
    assert (root/name).stat().st_size==size, name
print('FILE_NAMES_AND_SIZES_PASSED files=75 bytes=166107112571',flush=True)
def check(item):
    name,(wanted,size)=item
    path=root/name
    digest=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8*1024*1024): digest.update(chunk)
    assert digest.hexdigest()==wanted, ('HASH_MISMATCH',name)
    parquet=pq.ParquetFile(path)
    assert parquet.metadata.num_rows>0 and 'text' in parquet.schema_arrow.names,name
    return dict(file=name,bytes=size,rows=parquet.metadata.num_rows)
start=time.monotonic()
with ThreadPoolExecutor(max_workers=4) as executor:
    rows=[]
    for result in executor.map(check,expected.items()):
        rows.append(result)
        print('VERIFIED',json.dumps(result),flush=True)
counts=collections.Counter(Path(row['file']).parts[0] for row in rows)
assert counts==dict(en=50,vi=5,zh=5,ru=5,de=5,ar=5),counts
print('CULTURAX_DOWNLOAD_VERIFIED',json.dumps(dict(success=True,root=str(root),
    files=len(rows),bytes=sum(x['bytes'] for x in rows),languages=dict(counts),
    elapsed_seconds=round(time.monotonic()-start,2))),flush=True)
PY
date -u
echo CULTURAX_DOWNLOAD_HASH_AND_PARQUET_CHECK_COMPLETE
/mnt/local/conda-py311/envs/train_env/bin/python -B - <<'PY'
import os
from pathlib import Path
model=Path('/mnt/local/_models/deep-llms_th2/Qwen3-0.6B')
for name in ('config.json','tokenizer.json','tokenizer_config.json'):
    p=model/name
    print('TOKENIZER_FILE',str(p),p.stat().st_size if p.exists() else 'ABSENT')
out=Path('/mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B')
print('SAMPLING_OUTPUT_EXISTS',out.exists())
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        argv=(proc/'cmdline').read_bytes().split(b'\0')
        if argv and b'python' in Path(os.fsdecode(argv[0])).name.encode() and any(x.endswith(b'prepare_data.py') for x in argv):
            print('EXISTING_SAMPLER',proc.name,[os.fsdecode(x) for x in argv if x])
    except (FileNotFoundError,ProcessLookupError,PermissionError): pass
PY
df -h /mnt/local
free -h
echo SAMPLING_PREREQUISITE_CHECK_COMPLETE
