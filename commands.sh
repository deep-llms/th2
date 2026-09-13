#1 +30+a
#th2-ccm-verify-burn-and-package-results-20260913-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-8mgy-worker-0
test "$(tmux display-message -p -t ccm_burn_20260913_a01 '#{pane_dead}')" = 0
TASK_OUTPUT=/mnt/local/_outputs/@PROJECT@/ccm_smoke_20260913_a01
test ! -e "$TASK_OUTPUT/post_handoff_burn.json"
/usr/bin/python3 scripts/ccm_smoke_gpu_control.py verify-burn > "$TASK_OUTPUT/post_handoff_burn.json"
sleep 10
/usr/bin/python3 scripts/ccm_smoke_gpu_control.py verify-burn
/mnt/local/conda-py311/envs/train_env/bin/python - "$TASK_OUTPUT" <<'PY'
import hashlib,io,json,tarfile,sys
from pathlib import Path
out=Path(sys.argv[1])
assert json.loads((out/'payload_verified.json').read_text())['success']
files=['data_validation.json','coverage.json','payload_verified.json','burn_verification.json',
       'post_handoff_burn.json','payload.log','burn.log','common/train.jsonl','common/run.json',
       'common/complete.json','common/checkpoint-8/checkpoint.json','common/checkpoint-8/config.json',
       'performance/complete.json']
files += [f'pipeline_gpu{i}.log' for i in range(8)]
files += [f'performance/{phase}_{arm}.json' for phase,arm in
          [('common','base'),('stage1','contextual'),('stage1','grad'),('stage2','contextual'),('stage2','grad')]]
for i in range(8):
    assert 'CUDA OFFLINE PIPELINE PASS' in (out/f'pipeline_gpu{i}.log').read_text()
perf=json.loads((out/'performance/complete.json').read_text())
assert perf['success'] and len(perf['cases'])==5
payload={}
for name in files:
    p=out/name
    assert p.is_file() and not p.is_symlink(),name
    payload[name]=p.read_bytes()
data=Path('/mnt/local/_data/deep-llms_th2/ccm/smoke_20260913_a01')
payload['inputs/corpus_manifest.json']=(data/'corpus/manifest.json').read_bytes()
payload['inputs/vocabulary_metadata.json']=(data/'vocabulary.npz.json').read_bytes()
payload['inputs/base_assets.json']=Path('resources/qwen3_base_assets.json').read_bytes()
payload['inputs/culturax_raw_manifest.tsv']=Path('resources/culturax_raw_manifest.tsv').read_bytes()
manifest={name:dict(bytes=len(buf),sha256=hashlib.sha256(buf).hexdigest()) for name,buf in payload.items()}
payload['SHA256_MANIFEST.json']=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
archive=out/'ccm_smoke_results_20260913.tar.gz'
with archive.open('xb') as raw:
    with tarfile.open(fileobj=raw,mode='w:gz') as tar:
        for name,buf in payload.items():
            info=tarfile.TarInfo(name); info.size=len(buf)
            tar.addfile(info,io.BytesIO(buf))
with archive.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
with (out/'ccm_smoke_results_20260913.sha256').open('x') as f:
    f.write(digest+'  '+archive.name+'\n')
print('RESULT_ARCHIVE',archive,archive.stat().st_size,digest,flush=True)
for c in perf['cases']:
    print('PERFORMANCE',c['phase'],c['arm'],'tokens_per_second',c['post_warmup_input_tokens_per_second'],
          'peak_GiB',c['peak_allocated_bytes_all_ranks']/2**30,flush=True)
print('CCM_RESULTS_PACKAGED_AND_PERSISTENT_BURNS_VERIFIED',flush=True)
PY
date -u
