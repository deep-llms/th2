"""Small new 512-token input from the authenticated heal tail beyond old E2."""
import json
from pathlib import Path
import numpy as np
from .data import sha


def select_rows(ids, starts, first_token, stop_token, needed, length=512):
    if first_token % length or stop_token % length or stop_token <= first_token:
        raise ValueError('Aligned nonempty source interval required')
    if ids.shape != starts.shape or ids.ndim != 2 or ids.shape[1] != length:
        raise ValueError('Source shape mismatch')
    first, stop = first_token//length, stop_token//length
    if stop > len(ids):raise ValueError('Source interval beyond corpus')
    # Exclude the document straddling the old-prefix boundary, even when its
    # remaining rows contain no internal document start and look eligible.
    boundary = np.flatnonzero(starts[first:stop].reshape(-1))
    if not len(boundary):raise ValueError('No complete new document begins in interval')
    first_new_token = first_token + int(boundary[0])
    first_row = (first_new_token + length - 1)//length
    pool=[]
    for lo in range(first_row,stop,8192):
        b=starts[lo:min(lo+8192,stop)]
        pool.extend((np.flatnonzero(~b[:,1:].any(1))+lo).tolist())
    pool=np.asarray(pool,dtype=np.int64)
    if len(pool)<needed:raise ValueError(f'Insufficient new tail rows: {len(pool)} < {needed}')
    selected=np.random.default_rng(19220).choice(pool,needed,replace=False)
    if selected.min()*length < first_new_token or np.any(starts[selected,1:]):
        raise ValueError('New interval selection crosses document/prefix')
    return selected,dict(first_new_document_token=first_new_token,first_row=first_row,
        stop_row=stop,eligible_rows=len(pool),selected=needed,
        prefix_token_overlap=0,prefix_source_document_overlap=0)


def prepare_extension(spec,out,rows_per_seed,length=512):
    out=Path(out)
    manifest=json.loads(Path(spec['manifest']).read_text())
    prefix=json.loads(Path(spec['prefix_manifest']).read_text())
    if prefix != spec['prefix_expected']:raise ValueError('Prefix manifest differs from verified reconstruction')
    if prefix['source_manifest_sha256'] != sha(spec['manifest']):raise ValueError('Prefix provenance changed')
    if prefix['streams']['train']['tokens'] != spec['first_token']:raise ValueError('Old prefix size changed')
    if prefix['tokenizer_sha256'] != manifest['tokenizer_sha256'] or prefix['partition'] != manifest['partition']:
        raise ValueError('Tokenizer or held-out partition changed')
    source=spec['source_stem']
    ids=np.memmap(source+'.bin',mode='r',dtype='<u4').reshape(-1,length)
    starts=np.memmap(source+'.starts',mode='r',dtype='u1').reshape(-1,length)
    selected,proof=select_rows(ids,starts,spec['first_token'],spec['stop_token'],2*rows_per_seed,length)
    stem=str(out/'recovery_train')
    with open(stem+'.bin','xb') as ft,open(stem+'.starts','xb') as fs:
        for lo in range(0,len(selected),4096):
            ix=selected[lo:lo+4096];a=np.array(ids[ix]);b=np.array(starts[ix]);b[:,0]=1
            a.tofile(ft);b.tofile(fs)
    np.save(out/'source_rows.npy',selected)
    proof.update(source_stem=source,source_manifest_sha256=sha(spec['manifest']),
        prefix_manifest_sha256=sha(spec['prefix_manifest']),
        source_rows_sha256=sha(out/'source_rows.npy'),
        output_sha256={stem+s:sha(stem+s) for s in ('.bin','.starts')},
        note='Disjoint source-document instances beyond old prefix; not corpus-wide text deduplication')
    return stem,proof
