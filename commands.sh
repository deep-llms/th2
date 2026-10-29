#1 +60+a
#th2-6y3o-a2-fwe-final-verify-20261029-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
cd /mnt/local/_outputs/deep-llms_th2/6y3o-a2-corpora-20261029-a03/source
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PYCODE'
import hashlib,json,socket,subprocess
from pathlib import Path
from datetime import datetime,timezone
from scripts.a2_prepare_corpora import sha,verify_packed,tokenizer_identity,budgets
from efficient_llm.a2.data import PackedRows
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
source=Path.cwd();launch=source.parent
sm=json.loads((launch/'source-manifest.json').read_text())
assert sm['code_commit'].startswith('5fdf313') and sm['cpu_only']
assert len(sm['files'])==5
for rel,h in sm['files'].items():assert sha(source/rel)==h
assert sha(source/'plan.json')==sm['plan_sha256']
plan=json.loads((source/'plan.json').read_text());data=Path(plan['root'])
assert not (data/'failure.json').exists()
complete=json.loads((data/'complete.json').read_text())
assert complete['status']=='complete' and complete['plan_sha256']==sm['plan_sha256']
out=data/'packed/fineweb_edu';manifest=json.loads((out/'manifest.json').read_text())
assert manifest['status']=='complete' and manifest['corpus']=='fineweb_edu'
assert manifest['plan_sha256']==sm['plan_sha256'] and manifest['eos_id']==0
assert manifest['tokenizer_sha256']==tokenizer_identity(plan)
assert complete['corpora']['fineweb_edu']['status']=='complete'
assert complete['corpora']['fineweb_edu']['manifest_sha256']==sha(out/'manifest.json')
assert complete['corpora']['fineweb_edu']['streams']==manifest['streams']
assert {g:r['tokens'] for g,r in manifest['streams'].items()}==budgets(plan)
assert budgets(plan)==dict(heal=29999996928,tuning=4194304,final=16777216)
records=plan['corpora']['fineweb_edu']['files']
receipts={}
for g,target in budgets(plan).items():
    entries=[r for r in manifest['source_prefixes'] if r['group']==g]
    assert [r['source_index'] for r in entries]==list(range(len(entries)))
    assert sum(r['tokens'] for r in entries)==target
    remaining=target
    for r in entries:
        part=data/'parts/fineweb_edu'/f"{r['source_index']:05d}"/'complete.json'
        assert sha(part)==r['shard_receipt_sha256']
        saved=json.loads(part.read_text())
        assert saved['status']=='complete' and saved['source']==records[r['source_index']]
        assert saved['tokenizer_sha256']==manifest['tokenizer_sha256']
        assert r['tokens']==min(remaining,saved['streams'][g]['tokens'])
        remaining-=r['tokens'];receipts[str(r['source_index'])]=sha(part)
print('FWE_FINAL_VERIFICATION_STARTED',datetime.now(timezone.utc).isoformat(),flush=True)
verify_packed(out,manifest)
for g in ('heal','tuning','final'):
    ds=PackedRows(out,g,count=8,verify=False)
    assert len(ds[0]['input_ids'])==4096
session=subprocess.run(['tmux','display-message','-p','-t','a2_corpora_20261029_a03','#{pane_dead} #{pane_dead_status}'],capture_output=True,text=True,check=True).stdout.strip()
assert session=='1 0',session
receipt=dict(at=datetime.now(timezone.utc).isoformat(),status='verified',corpus='fineweb_edu',manifest_sha256=sha(out/'manifest.json'),complete_sha256=sha(data/'complete.json'),source=sm,streams=manifest['streams'],source_receipts=receipts,native_reader_passed=True,full_output_hashes_passed=True,session=session)
print('FWE_FINAL_VERIFIED',json.dumps(receipt),flush=True)
for name,p in [('fwe_manifest.json',out/'manifest.json'),('corpora_complete.json',data/'complete.json')]:
    raw=p.read_bytes();print('ARTIFACT',json.dumps(dict(path=name,sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
PYCODE
