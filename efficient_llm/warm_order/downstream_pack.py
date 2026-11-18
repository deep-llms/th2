"""Lossless numeric result export, excluding redundant raw score/harness JSON."""
import json,shutil,tarfile,time
from pathlib import Path
import numpy as np
from .data import sha
from .train import write_json
from .downstream_report import TASKS
from scripts.gpu_process import inspect,ownership

def pack_items(items,path):
    arrays={};schema={}
    for task,x in items.items():
        rows=x['items'];n=len(rows);width=max(len(r['ll']) for r in rows)
        def put(key,v):arrays[task+'_'+key]=v
        put('id',np.array([r['id'] for r in rows],dtype=np.int64))
        put('identity',np.array([list(bytes.fromhex(r['identity'])) for r in rows],dtype=np.uint8))
        put('count',np.array([len(r['ll']) for r in rows],dtype=np.int16))
        for key,dtype in [('ll',np.float64),('lengths',np.int32)]+([] if task=='lambada_openai' else [('chars',np.int32)]):
            a=np.zeros((n,width),dtype=dtype)
            for i,r in enumerate(rows):a[i,:len(r[key])]=r[key]
            put(key,a)
        for key in (['correct','greedy'] if task=='lambada_openai' else ['correct','acc','acc_norm','gold']):put(key,np.array([r[key] for r in rows],dtype=np.int16))
        schema[task]=dict(metric=x['metric'],accuracy=x['accuracy'],items=n)
    np.savez_compressed(path,**arrays)
    return schema

def finalize():
    base=Path('/mnt/local/_outputs/deep-llms_th2/warm_downstream_replay_20261118_a01')
    out=base/'handoff-a01'
    if out.exists():raise ValueError('Existing handoff; retrieve rather than repeat')
    plan=json.loads((base/'plan.json').read_text());rec=json.loads((base/'recovery.json').read_text())
    state=inspect()
    if state['guard_disabled'] or ownership(state['gpus'])!=ownership(rec['ownership']['gpus']):raise ValueError('Native ownership changed')
    for p,w in json.loads((base/'recovery-source.json').read_text()).items():
        if sha(base/'recovery-source'/p)!=w:raise ValueError('Recovery source changed')
    out.mkdir()
    for rel in ['REPORT.json','REPORT.md','RESULTS.csv','plan.json','gpu-probe.json','execution/source.json','execution/admission.json','recovery.json','recovery-source.json']:
        dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base/rel,dst)
    for e in plan['models']:
        run=base/'runs'/e['name'];target=out/'runs'/e['name'];target.mkdir(parents=True)
        done=json.loads((run/'complete.json').read_text());replay=json.loads((run/'replay-complete.json').read_text())
        for rel,w in done['files'].items():
            if sha(run/rel)!=w:raise ValueError('Score bytes changed')
        if sha(run/'items.json')!=replay['items_sha256'] or sha(run/'raw.json')!=replay['raw_sha256']:raise ValueError('Replay bytes changed')
        schema=pack_items(json.loads((run/'items.json').read_text()),target/'items.npz')
        write_json(target/'items_schema.json',schema)
        for rel in ('details.json','complete.json','replay-complete.json'):shutil.copyfile(run/rel,target/rel)
    write_json(out/'pack.json',dict(format='lossless_numeric_v1',original=str(base),raw_scores_and_harness_json='node_only; all task continuation LL retained losslessly in NPZ',ownership=state,utc=time.time(),source_sha256=sha(__file__)))
    files=sorted(p for p in out.rglob('*') if p.is_file())
    with tarfile.open(out/'results.tar.gz','w:gz') as t:
        for p in files:t.add(p,arcname=str(p.relative_to(out)),recursive=False)
    archive=out/'results.tar.gz'
    if archive.stat().st_size>25_000_000:raise ValueError('Compact export still above cap; do not pull')
    backup=Path('/mnt/local/_backup/deep-llms_th2/warm_downstream_handoff_20261118_a01');backup.mkdir()
    shutil.copyfile(archive,backup/'results.tar.gz')
    if sha(backup/'results.tar.gz')!=sha(archive):raise ValueError('Backup mismatch')
    proof=dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files={str(p.relative_to(out)):sha(p) for p in files},node_backup=str(backup/'results.tar.gz'),weights_exported=False)
    write_json(out/'backup.json',proof);write_json(backup/'backup.json',proof)
    final=inspect()
    if final['guard_disabled'] or ownership(final['gpus'])!=ownership(state['gpus']):raise ValueError('Native ownership changed during packaging')
    terminal=dict(success=True,utc=time.time(),no_gpu_work=True,native_unchanged=True,models=12,rows=48,archive_sha256=proof['archive_sha256'],bytes=proof['bytes'],proof_sha256=sha(out/'backup.json'),proof_bytes=(out/'backup.json').stat().st_size,members=len(files),node_backup=proof['node_backup'])
    write_json(out/'terminal.json',terminal)
    print('DOWNSTREAM_HANDOFF',json.dumps(terminal),flush=True)

if __name__=='__main__':finalize()
