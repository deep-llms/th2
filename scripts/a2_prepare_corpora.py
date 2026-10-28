"""Offline CPU preparation of pinned DCLM/FineWeb-Edu corpora for A2.

No network or GPU work. Completed source shards are reusable after hash checks;
partial shard directories are preserved and refused, never silently overwritten.
"""
import argparse
import fcntl
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import struct
import time

GROUPS = ('heal', 'tuning', 'final')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def group_for(text):
    digest = hashlib.sha256(text.encode('utf8')).digest()
    bucket = int.from_bytes(digest[:8], 'big') % 10000
    return ('tuning' if bucket < 2 else 'final' if bucket < 10 else 'heal'), digest


def verify_file(path, expected):
    path = Path(path)
    if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
        raise ValueError('File identity differs: ' + str(path))


def wait_source(record, deadline):
    path = Path(record['local'])
    while not path.is_file() or path.stat().st_size != record['bytes']:
        if time.monotonic() >= deadline: raise TimeoutError('Download not ready: ' + str(path))
        time.sleep(30)
    verify_file(path, record)
    return path


def tokenizer_identity(plan):
    base = Path(plan['tokenizer_path'])
    for name, expected in plan['tokenizer_files'].items():
        if sha(base / name) != expected: raise ValueError('Tokenizer hash differs: ' + name)
    return plan['tokenizer_files']['tokenizer.json']


def shard_worker(task):
    record, tokenizer_file, tokenizer_sha, output, deadline = task
    import numpy as np
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    output = Path(output)
    receipt = output / 'complete.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['source'] != record or saved['tokenizer_sha256'] != tokenizer_sha:
            raise ValueError('Existing shard belongs to different inputs')
        for name, info in saved['files'].items(): verify_file(output / name, info)
        return saved
    raw = wait_source(record, deadline)
    output.mkdir(parents=True, exist_ok=False)
    if sha(tokenizer_file) != tokenizer_sha: raise ValueError('Tokenizer changed')
    tok = Tokenizer.from_file(str(tokenizer_file)); tok.no_padding(); tok.no_truncation()
    if tok.token_to_id('<|endoftext|>') != 0: raise ValueError('Expected EOS0')
    handles = {g: ((output / (g+'.bin')).open('xb'), (output / (g+'.starts')).open('xb')) for g in GROUPS}
    index = (output / 'documents.bin').open('xb')
    counts = {g: dict(documents=0,tokens=0) for g in GROUPS}
    rows = empty = 0; started = time.monotonic()
    try:
        for batch in pq.ParquetFile(raw).iter_batches(batch_size=64,columns=['text'],use_threads=False):
            if time.monotonic() >= deadline: raise TimeoutError('Preparation deadline reached')
            texts = batch.column(0).to_pylist()
            if any(not isinstance(t,str) for t in texts): raise ValueError('Non-string text')
            encodings = tok.encode_batch(texts,add_special_tokens=False)
            for text, encoding in zip(texts, encodings, strict=True):
                row = rows; rows += 1
                if not text.strip(): empty += 1; continue
                group, digest = group_for(text)
                ids = np.asarray(encoding.ids+[0],dtype='<u4')
                if ids.max() >= tok.get_vocab_size(): raise ValueError('Token outside vocabulary')
                starts = np.zeros(len(ids),dtype='u1'); starts[0] = 1
                handles[group][0].write(ids.tobytes()); handles[group][1].write(starts.tobytes())
                counts[group]['documents'] += 1; counts[group]['tokens'] += len(ids)
                # Fixed 49-byte records: source row, exact-text SHA256, token count, group index.
                index.write(struct.pack('<Q32sQB',row,digest,len(ids),GROUPS.index(group)))
    finally:
        index.close()
        for pair in handles.values():
            for f in pair: f.close()
    files = {p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir() if p.is_file()}
    saved = dict(status='complete',source=record,tokenizer_sha256=tokenizer_sha,rows=rows,
                 empty_documents_skipped=empty,streams=counts,files=files,seconds=time.monotonic()-started)
    write_json(receipt,saved)
    print('CORPUS_SHARD_COMPLETE',json.dumps(dict(index=record['index'],tokens=sum(v['tokens'] for v in counts.values()),seconds=saved['seconds'])),flush=True)
    return saved


def budgets(plan):
    n = plan['length']
    return dict(heal=plan['train_tokens']//n*n,tuning=plan['tuning_rows']*n,final=plan['final_rows']*n)


def copy_prefix(source, dest, size):
    remaining = size
    with Path(source).open('rb') as f:
        while remaining:
            b = f.read(min(8 << 20,remaining))
            if not b: raise ValueError('Short shard stream')
            dest.write(b); remaining -= len(b)


def assemble(parts, output, plan, corpus, plan_sha):
    import numpy as np
    output = Path(output); targets = budgets(plan)
    if output.exists():
        manifest = json.loads((output/'manifest.json').read_text())
        if manifest['plan_sha256'] != plan_sha or manifest['corpus'] != corpus: raise ValueError('Output identity differs')
        verify_packed(output,manifest); return manifest
    output.mkdir(parents=True,exist_ok=False)
    streams = {}; provenance = []
    for g in GROUPS:
        count = 0
        with (output/(g+'.bin')).open('xb') as ids, (output/(g+'.starts')).open('xb') as starts:
            for root, saved in parts:
                take = min(targets[g]-count,saved['streams'][g]['tokens'])
                if take:
                    for suffix in ('.bin','.starts'): verify_file(root/(g+suffix),saved['files'][g+suffix])
                    copy_prefix(root/(g+'.bin'),ids,take*4); copy_prefix(root/(g+'.starts'),starts,take)
                    provenance.append(dict(group=g,source_index=saved['source']['index'],tokens=take,
                        shard_receipt_sha256=sha(root/'complete.json')))
                    count += take
                if count == targets[g]: break
        if count != targets[g]: raise ValueError('Insufficient source tokens for '+g)
        streams[g] = dict(rows=count//plan['length'],tokens=count,length=plan['length'],
            token_sha256=sha(output/(g+'.bin')),starts_sha256=sha(output/(g+'.starts')))
    manifest = dict(status='complete',schema='a2-new-corpus-v1',corpus=corpus,plan_sha256=plan_sha,
        tokenizer_sha256=plan['tokenizer_files']['tokenizer.json'],eos_id=0,streams=streams,
        small_indices=sorted(np.random.default_rng(2029).choice(plan['tuning_rows'],min(256,plan['tuning_rows']),replace=False).tolist()),
        source_prefixes=provenance,partition='SHA256(exact UTF8 text), first8bytes big-endian mod10000: tuning0..1, final2..9, heal10..9999',
        packing='continuous across documents/shards; no BOS; append EOS0; cut only at final full-sequence budget',
        caveat='May overlap parent pretraining. Distinct corpus; not the old CulturaX Phase0A protocol.')
    verify_packed(output,manifest)
    write_json(output/'manifest.json',manifest)
    return manifest


def verify_packed(output, manifest):
    output = Path(output)
    for name, info in manifest['streams'].items():
        if info['tokens'] != info['rows']*info['length']: raise ValueError('Invalid sequence count')
        verify_file(output/(name+'.bin'),dict(bytes=info['tokens']*4,sha256=info['token_sha256']))
        verify_file(output/(name+'.starts'),dict(bytes=info['tokens'],sha256=info['starts_sha256']))


def prepare(plan_path, deadline_hours=48):
    plan_path = Path(plan_path); plan = json.loads(plan_path.read_text()); plan_sha = sha(plan_path)
    root = Path(plan['root']); root.mkdir(parents=True,exist_ok=True)
    lock = (root/'.prepare.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    tokenizer_sha = tokenizer_identity(plan)
    if plan['eos_id'] != 0 or plan['length'] != 4096: raise ValueError('Wrong tokenization protocol')
    # Retain raw data and shard staging; do not delete unrelated or old artifacts.
    if shutil.disk_usage(root).free < 1_000_000_000_000: raise ValueError('Need at least 1TB free before two-corpus preparation')
    deadline = time.monotonic()+deadline_hours*3600; started = time.monotonic()
    status = dict(status='running',plan_sha256=plan_sha,corpora={})
    write_json(root/'status.json',status)
    os.environ['TOKENIZERS_PARALLELISM']='false'; os.environ['RAYON_NUM_THREADS']='1'
    os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'
    for name, corpus in plan['corpora'].items():
        selected = []; totals = dict.fromkeys(GROUPS,0); targets = budgets(plan)
        workers = min(corpus['workers'],max(1,len(os.sched_getaffinity(0))//2))
        files = corpus['files']
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for start in range(0,len(files),workers):
                batch = files[start:start+workers]
                tasks = [(f,Path(plan['tokenizer_path'])/'tokenizer.json',tokenizer_sha,root/'parts'/name/f"{f['index']:05d}",deadline) for f in batch]
                for task, result in zip(tasks,pool.map(shard_worker,tasks),strict=True):
                    selected.append((Path(task[3]),result))
                    for g in GROUPS: totals[g] += result['streams'][g]['tokens']
                status['corpora'][name]=dict(status='tokenizing',shards=len(selected),tokens=totals.copy(),elapsed_seconds=time.monotonic()-started)
                write_json(root/'status.json',status);print('CORPUS_PROGRESS',json.dumps(status),flush=True)
                if all(totals[g]>=targets[g] for g in GROUPS): break
        if not all(totals[g]>=targets[g] for g in GROUPS): raise ValueError('Pinned source subset insufficient; extend manifest explicitly, do not repeat data')
        manifest = assemble(selected,root/'packed'/name,plan,name,plan_sha)
        # Native reader compatibility after full digest checks, without loading GPU models.
        from efficient_llm.a2.data import PackedRows
        for group in GROUPS:
            ds=PackedRows(root/'packed'/name,group,count=min(8,manifest['streams'][group]['rows']),verify=False)
            if len(ds[0]['input_ids'])!=4096: raise ValueError('Native PackedRows mismatch')
        status['corpora'][name]=dict(status='complete',manifest_sha256=sha(root/'packed'/name/'manifest.json'),streams=manifest['streams'])
        write_json(root/'status.json',status)
    status.update(status='complete',elapsed_seconds=time.monotonic()-started)
    write_json(root/'status.json',status);write_json(root/'complete.json',status)
    print('CORPORA_COMPLETE',json.dumps(status),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--deadline-hours',type=float,default=48);a=p.parse_args()
    try: prepare(a.plan,a.deadline_hours)
    except Exception as e:
        root=Path(json.loads(a.plan.read_text())['root'])
        if root.is_dir(): write_json(root/'failure.json',dict(error=type(e).__name__,message=str(e)))
        raise


if __name__=='__main__': main()
