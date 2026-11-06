"""Offline seeded mixture preparation with immutable raw files and document traces."""
import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import math
import time
from pathlib import Path
import numpy as np
from efficient_llm.a2_family.common import read,write,sha,SPEC_SHA
from scripts.a2_prepare_corpora import verify_file


def quotas(weights,total):
    names=sorted(weights);normal=sum(weights.values())
    exact={n:total*weights[n]/normal for n in names}
    result={n:math.floor(v) for n,v in exact.items()}
    for n in sorted(names,key=lambda n:(-(exact[n]-result[n]),n))[:total-sum(result.values())]:result[n]+=1
    return result


def wait_verified(record,progress):
    p=Path(record['local']);p=p/Path(record['path']).name if p.is_dir() else p
    while not p.is_file() or p.stat().st_size!=record['bytes']:
        write(progress,dict(state='waiting_raw',path=str(p),expected_bytes=record['bytes']))
        time.sleep(30)
    verify_file(p,record)
    return p


def read_texts(files,columns):
    """Yield original raw rows in file order. No undocumented text normalization."""
    import pyarrow.parquet as pq
    for source,path in enumerate(files):
        if str(path).endswith('.parquet'):
            pf=pq.ParquetFile(path);available=pf.schema_arrow.names
            column=next((c for c in columns if c in available),None)
            if column is None:raise ValueError('No registered text column: '+str(path))
            row=0
            for batch in pf.iter_batches(batch_size=4096,columns=[column]):
                for text in batch.column(0).to_pylist():
                    if not isinstance(text,str):raise ValueError('Non-text document')
                    yield source,row,text,column;row+=1
        else:
            if str(path).endswith('.gz'):stream=gzip.open(path,'rt',encoding='utf8')
            elif str(path).endswith('.zst'):
                import zstandard as zstd
                stream=io.TextIOWrapper(zstd.open(path,'rb'),encoding='utf8')
            else:stream=path.open(encoding='utf8')
            with stream:
                for row,line in enumerate(stream):
                    doc=json.loads(line);column=next((c for c in columns if isinstance(doc.get(c),str)),None)
                    if column is None:raise ValueError('No registered JSON text field: '+str(path))
                    yield source,row,doc[column],column


def primary_exclusions(plan):
    root=Path(plan['primary_fwe_root'])
    while not (root/'complete.json').is_file():time.sleep(30)
    complete=read(root/'complete.json')
    if complete['status']!='complete':raise ValueError('Primary FWE incomplete')
    hashes=set();locks=[]
    for seed in ('2030','2031'):
        p=root/seed/'consumed_documents.jsonl';manifest=read(root/seed/'manifest.json')
        actual=sha(p)
        if actual!=manifest['streams']['heal']['exposure_sha256']:raise ValueError('Primary FWE trace hash changed')
        with p.open() as f:
            for line in f:hashes.add(json.loads(line)['text_sha256'])
        locks.append(dict(path=str(p),sha256=actual))
    return hashes,locks


def prepare_source(plan,entry,dest,target,exclusions):
    from tokenizers import Tokenizer
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    records=entry['files']
    if entry.get('node_plan_override'):
        p=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-data-20261104-a01/corpus_plan.json')
        if sha(p)!='b39fd4953a3c078945a56f33a5c97da9ed14966723bee633f27dc9445b12de01':raise ValueError('Historical FWE raw plan changed')
        records=read(p)['corpora']['fineweb_edu']['files']
    files=[wait_verified(r,dest/'progress.json') for r in records]
    tok=Tokenizer.from_file(plan['tokenizer']['local']);eos=plan['eos_id']
    # Build a disk-backed text index. This permits an exactly uniform permutation
    # of every document in the declared source pool without holding texts in RAM.
    index=[];fields=set();rejected=0;raw_docs=0;pool_overlaps=[]
    with (dest/'text.utf8').open('xb') as storage:
        for source,row,text,column in read_texts(files,entry['text_columns']):
            raw_docs+=1;encoded=text.encode('utf8');digest=hashlib.sha256(encoded).hexdigest();fields.add(column)
            if digest in plan.get('heldout_hashes',{}):pool_overlaps.append(dict(raw_file=source,raw_row=row,text_sha256=digest,heldout=plan['heldout_hashes'][digest],excluded_primary=digest in exclusions))
            if entry['name'] in ('fineweb-edu','dclm') and digest in exclusions:rejected+=1;continue
            index.append((storage.tell(),len(encoded),source,row));storage.write(encoded)
            if raw_docs%100000==0:write(dest/'progress.json',dict(stage='indexing',raw_documents=raw_docs,excluded_primary=rejected))
    arr=np.asarray(index,dtype='<u8');np.save(dest/'text_index.npy',arr);del index
    if not len(arr):raise ValueError('No eligible documents')
    order=np.random.default_rng(2030).permutation(len(arr));np.save(dest/'document_order.npy',order)
    count=documents=0;overlaps=[]
    heldout=plan.get('heldout_hashes',{})
    with (dest/'text.utf8').open('rb') as raw,(dest/'tokens.bin').open('xb') as output,(dest/'documents.jsonl').open('x') as trace:
        for start in range(0,len(order),64):
            indices=order[start:start+64];texts=[]
            for i in indices:
                off,size,_,_=map(int,arr[int(i)]);raw.seek(off);texts.append(raw.read(size).decode('utf8'))
            tokenized=tok.encode_batch(texts,add_special_tokens=False)
            for i,text,ids in zip(indices,texts,tokenized,strict=True):
                _,_,source,row=map(int,arr[int(i)]);values=np.asarray(ids.ids+[eos],dtype='<u4');take=min(len(values),target-count)
                output.write(values[:take].tobytes());h=hashlib.sha256(text.encode('utf8')).hexdigest()
                record=dict(pool_index=int(i),raw_file=source,raw_row=row,text_sha256=h,source=entry['name'],source_start=count,source_stop=count+take,full_document_tokens=len(values),consumed_tokens=take)
                trace.write(json.dumps(record)+'\n');count+=take;documents+=1
                if h in heldout:overlaps.append(dict(record,heldout=heldout[h]))
                if count==target:break
            write(dest/'progress.json',dict(stage='tokenizing',tokens=count,target=target,documents=documents))
            if count==target:break
    if count!=target:raise ValueError(f'Source pool insufficient: {entry["name"]} {count}/{target}; preserve artifacts and expand registered input pool')
    return write(dest/'complete.json',dict(status='complete',name=entry['name'],tokens=count,documents=documents,eligible_documents=len(arr),raw_documents=raw_docs,excluded_primary=rejected,seed=2030,text_fields=sorted(fields),
            token_file=str(dest/'tokens.bin'),token_sha256=sha(dest/'tokens.bin'),documents_path=str(dest/'documents.jsonl'),documents_sha256=sha(dest/'documents.jsonl'),document_order_sha256=sha(dest/'document_order.npy'),
            raw_files=[dict(path=str(p),sha256=r['sha256'],bytes=r['bytes']) for p,r in zip(files,records,strict=True)],consumed_overlap=overlaps,pool_overlap=pool_overlaps))


def interleave(sources,weights,dest,total):
    """Token-balanced interleaving at real document boundaries; keep all traces."""
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False);names=sorted(sources)
    used={n:0 for n in names};position=0;prefix=762*64*4096;prefix_used={n:0 for n in names}
    handles={n:Path(sources[n]['token_file']).open('rb') for n in names}
    traces={n:Path(sources[n]['documents_path']).open() for n in names}
    try:
        with (dest/'heal.bin').open('xb') as output,(dest/'heal.starts').open('xb') as starts,(dest/'consumed_documents.jsonl').open('x') as trace:
            while position<total:
                available=[n for n in names if used[n]<sources[n]['tokens']]
                if not available:raise ValueError('Insufficient source tokens')
                name=min(available,key=lambda n:(used[n]/weights[n],n))
                line=traces[name].readline()
                if not line:raise ValueError('Source trace ended early')
                doc=json.loads(line);take=doc['consumed_tokens'];raw=handles[name].read(take*4)
                if len(raw)!=take*4 or position+take>total:raise ValueError('Source trace/token counts disagree')
                output.write(raw);boundary=np.zeros(take,dtype='u1');boundary[0]=1;starts.write(boundary.tobytes())
                prefix_used[name]+=max(0,min(position+take,prefix)-min(position,prefix))
                trace.write(json.dumps(dict(doc,stream_start=position,stream_stop=position+take))+'\n')
                used[name]+=take;position+=take
    finally:
        for f in [*handles.values(),*traces.values()]:f.close()
    shares={n:used[n]/total for n in names};prefix_shares={n:prefix_used[n]/min(total,prefix) for n in names}
    checks=[dict(test='FT8_shares_full',passed=all(abs(shares[n]-weights[n])<=.01*weights[n] for n in names)),
            dict(test='FT8_shares_0.2B',passed=all(abs(prefix_shares[n]-weights[n])<=.01*weights[n] for n in names))]
    info=dict(rows=total//4096,tokens=total,length=4096,token_sha256=sha(dest/'heal.bin'),starts_sha256=sha(dest/'heal.starts'),exposure_sha256=sha(dest/'consumed_documents.jsonl'))
    return write(dest/'manifest.json',dict(status='complete',streams={'heal':info},weights=weights,token_counts=used,shares=shares,prefix_0_2B_shares=prefix_shares,seed=2030,checks=checks,share_tolerance='1 percent relative to source weight (conservative interpretation)',packing='whole documents interleaved by token deficit; explicit EOS; document starts retained; no BOS'))


def prepare(plan_path,output,cx_root):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);plan=read(plan_path)
    if plan['spec_sha256']!=SPEC_SHA:raise ValueError('Wrong mixture spec')
    wait_verified(dict(plan['tokenizer'],path='tokenizer.json'),root/'tokenizer_progress.json')
    excluded,locks=primary_exclusions(plan);plan['heldout_hashes']={}
    for group in ('tuning','final'):
        for i,d in enumerate(read(Path(cx_root)/group/'documents.json')['documents']):plan['heldout_hashes'].setdefault(d['text_sha256'],[]).append(dict(set=group,index=i))
    weights={e['name']:e['effective_weight'] for e in plan['sources']};targets=quotas(weights,plan['prepared_tokens']);ready={};failed={}
    def task(entry):return prepare_source(plan,entry,root/'sources'/entry['name'],targets[entry['name']],excluded)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks={pool.submit(task,e):e['name'] for e in plan['sources']}
        for future in concurrent.futures.as_completed(tasks):
            name=tasks[future]
            try:ready[name]=future.result()
            except Exception as error:failed[name]=repr(error)
            write(root/'progress.json',dict(ready=ready,failed=failed,source_targets=targets))
    # Actual corruption/insufficient files never become a claimed verified mixture.
    # A subsequent explicit preparation can redistribute truly unavailable sources.
    if failed:
        write(root/'complete.json',dict(status='incomplete',ready=ready,failed=failed));return
    values={n:read(v['path']) for n,v in ready.items()}
    stream=interleave(values,weights,root/'stream',plan['prepared_tokens'])
    write(root/'complete.json',dict(status='complete',stream=stream,sources=ready,primary_exclusions=locks,plan_sha256=sha(plan_path),flags=plan['flags'],missing_sources=plan['missing_sources']))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('plan','output','cx-root'):p.add_argument('--'+k,required=True)
    a=p.parse_args();prepare(a.plan,a.output,a.cx_root)
