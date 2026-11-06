"""CPU-only content-hash overlap and exact packed-token exposure audit."""
from pathlib import Path
import json
import numpy as np
from efficient_llm.a2_family.common import read,write,sha
from scripts.a2_stage1_data import INDEX
from scripts.a2_family_prepare_fwe import POOL_SHA


def exposures(trace,hashes):
    found=[]
    with Path(trace).open() as f:
        for line in f:
            d=json.loads(line)
            if d['text_sha256'] in hashes:found.append(d)
    return found


def audit(cfg,output):
    cx=Path(cfg['cx_root']);heldout={}
    for group in ('tuning','final'):
        info=read(cx/group/'manifest.json');path=cx/group/'documents.json'
        if sha(path)!=info['documents_sha256']:raise ValueError('Heldout metadata hash differs')
        for i,d in enumerate(read(path)['documents']):heldout.setdefault(d['text_sha256'],[]).append(dict(set=group,index=i,tokens=d['tokens'],predicted_tokens=d['predicted_tokens']))
    poolpath=Path(cfg['fwe_pool_documents'])
    if sha(poolpath)!=POOL_SHA:raise ValueError('Historical full FWE pool hash differs')
    pool=np.memmap(poolpath,dtype=INDEX,mode='r');targets=np.asarray([bytes.fromhex(h) for h in heldout],dtype='V32');matches=[]
    for start in range(0,len(pool),100000):
        chunk=pool[start:start+100000];indices=np.flatnonzero(np.isin(chunk['hash'],targets))
        for i in indices:
            d=chunk[i];h=bytes(d['hash']).hex();matches.append(dict(pool_index=start+int(i),text_sha256=h,heldout=heldout[h]))
    def summary(records,traces):
        hashes={r['text_sha256'] for r in records};indices={g:sorted({v['index'] for r in records for v in r['heldout'] if v['set']==g}) for g in ('tuning','final')}
        return dict(pool_overlaps=records,final_indices=indices['final'],tuning_indices=indices['tuning'],consumed_overlaps={s:exposures(p,hashes) for s,p in traces.items()},trace_locks={s:dict(path=str(p),sha256=sha(p)) for s,p in traces.items()})
    traces={s:Path(e['path'])/'consumed_documents.jsonl' for s,e in cfg['streams'].items()}
    for s,p in traces.items():
        if sha(p)!=read(p.parent/'manifest.json')['streams']['heal']['exposure_sha256']:raise ValueError('FWE exposure trace changed')
    result={'fwe':summary(matches,traces)}
    ready=read(cfg['mixture_complete']['path'])
    if sha(cfg['mixture_complete']['path'])!=cfg['mixture_complete']['sha256'] or ready['status']!='complete':raise ValueError('Mixture readiness changed')
    records=[]
    for name,lock in ready['sources'].items():
        if sha(lock['path'])!=lock['sha256']:raise ValueError('Mixture source audit changed')
        records.extend(dict(r,source=name) for r in read(lock['path'])['pool_overlap'])
    mp=Path(cfg['mixture_stream']['path'])/'consumed_documents.jsonl'
    if sha(mp)!=read(mp.parent/'manifest.json')['streams']['heal']['exposure_sha256']:raise ValueError('Mixture exposure trace changed')
    result['mix']=summary(records,{'2030':mp})
    write(output,dict(status='complete',datasets=result,fwe_pool_sha256=POOL_SHA,method='Whole declared source pools content hashes; actual run rows checked separately; common dropped evaluation-document indices per dataset'))
    return result
