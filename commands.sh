#1 +300+a
#th2-tjx3-document-stability-200-monitor-20261004-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python -u - <<'PY'
import hashlib,json,tarfile,time
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
from run_experiments import now
root=Path('/mnt/local/_outputs/deep-llms_th2/document-stability-200-20261004-a01')
for tick in range(60):
    if (root/'result.json').is_file():break
    print('READ_ONLY_PROGRESS',now(),flush=True)
    for mode in ['sdpa_isolated','fa4_isolated']:
        p=root/(mode+'.log')
        if p.is_file():
            lines=p.read_text().splitlines()
            losses=[line for line in lines if line.startswith("{'loss':")]
            print(mode,losses[-1] if losses else 'startup/evaluation',flush=True)
    time.sleep(30)
else:raise TimeoutError('No completion receipt; no processes were changed')
receipt=json.loads((root/'result.json').read_text())
print('FINAL_RESULT',json.dumps(receipt),flush=True)
status=inspect();print('LIVE_GPU_STATUS',json.dumps(status),flush=True)
if receipt.get('burn'):
    expected={g['index']:g['pids'] for g in receipt['burn']['gpus']}
    assert all(g['pids']==expected[g['index']] for g in status['gpus'])
    assert not status['guard_disabled']
paths=[root/'result.json',root/'benchmark/data.json',root/'benchmark/eval-data.json',
       root/'benchmark/correctness.json',root/'benchmark/summary.json',root/'benchmark/comparison.json']
paths += sorted((root/'benchmark').glob('*/rank-*.json'))
paths += sorted((root/'benchmark').glob('*/trainer_state.json'))
paths=[p for p in paths if p.is_file()]
# Copy the changing burn log once so archive and manifest see identical bytes.
p=root/'burn-snapshot.log';p.write_bytes((root/'burn.log').read_bytes());paths.append(p)
manifest={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
manifest_path=root/'result-files.json'
with manifest_path.open('x') as f:json.dump(manifest,f,indent=2)
archive=root/'result-export.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
    for p in paths+[manifest_path]:tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
print('RESULT_EXPORT',json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())),flush=True)
assert receipt.get('passed') and receipt.get('burn',{}).get('collective_progress_verified')
assert len(list((root/'benchmark').glob('*/rank-*.json')))==16
PY
