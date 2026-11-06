"""Retokenize exact historical FWE documents, retaining both full-pool permutations.

Only the required shuffled prefix is materialized. Candidate documents are an
initial prefix of each COMPLETE historical permutation, enlarged if necessary;
selection never changes the document pool, sequence, or tokenization.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scripts.a2_stage1_data import INDEX
from scripts.a2_prepare_corpora import verify_file
from efficient_llm.a2_family.common import read, write, sha, SPEC_SHA

POOL_SHA='82571448856cec259fcb6b29deb097a7671309f6d55d39f0321efb1ffc8d132e'
ORDER_SHA='685e403d261c34b35e237c71f9978a8d089399adfc0314ec3bee6dc66aff66ba'
MANIFEST_SHA='807e6807d6a39e4b7837c94e66723d2ffa7b3e62fc675cef357d2ed9e973071b'
TARGETS={2030:12207*64*4096,2031:762*64*4096}
PLAN_SHA='b39fd4953a3c078945a56f33a5c97da9ed14966723bee633f27dc9445b12de01'

def candidate_count(pool, order, old_token_target):
    sums=np.cumsum(pool['tokens'][order],dtype=np.uint64)
    return min(len(order),int(np.searchsorted(sums,old_token_target,side='left'))+1)

def emit_prefix(pool, order, offsets, token_file, dest, total, eos_id):
    """Write exact prefix plus per-document exposure records, including clipped last doc."""
    dest=Path(dest);dest.mkdir(exist_ok=False)
    tokens=np.memmap(token_file,dtype='<u4',mode='r');count=docs=0
    with (dest/'heal.bin').open('xb') as out,(dest/'heal.starts').open('xb') as boundary,(dest/'consumed_documents.jsonl').open('x') as trace:
        for pool_id in order:
            begin,length=map(int,offsets[int(pool_id)])
            if begin<0:raise ValueError('Selected prefix has an unencoded document')
            ids=tokens[begin:begin+length]
            if len(ids)!=length or int(ids[-1])!=eos_id:raise ValueError('Missing document EOS')
            take=min(length,total-count)
            out.write(ids[:take].tobytes());starts=np.zeros(take,dtype='u1');starts[0]=1;boundary.write(starts.tobytes())
            d=pool[int(pool_id)]
            trace.write(json.dumps(dict(pool_index=int(pool_id),source=int(d['source']),row=int(d['row']),
                text_sha256=bytes(d['hash']).hex(),stream_start=count,stream_stop=count+take,
                full_document_tokens=length,consumed_tokens=take))+'\n')
            count+=take;docs+=1
            if count==total:break
    if count!=total:raise ValueError('Prefix insufficient')
    return dict(rows=total//4096,tokens=total,length=4096,documents=docs,
                token_sha256=sha(dest/'heal.bin'),starts_sha256=sha(dest/'heal.starts'),
                exposure_sha256=sha(dest/'consumed_documents.jsonl'))

def prepare(plan_path, old_shuffle, downloads_path, output, data_inputs=None):
    from tokenizers import Tokenizer
    import pyarrow.parquet as pq
    plan_path=Path(plan_path);old=Path(old_shuffle);output=Path(output)
    if sha(plan_path)!=PLAN_SHA or sha(old/'manifest.json')!=MANIFEST_SHA:
        raise ValueError('Historical FWE manifest/plan identity changed')
    if sha(old/'documents.bin')!=POOL_SHA or sha(old/'document_order.bin')!=ORDER_SHA:
        raise ValueError('Historical pool/permutation identity changed')
    plan=read(plan_path);models=read(downloads_path)
    if models['spec_sha256']!=SPEC_SHA:raise ValueError('Wrong Family spec')
    tokenizer_entry=next(e for e in models['checkpoints'] if e['label']=='k6')
    tf=next(f for f in tokenizer_entry['files'] if f['name']=='tokenizer.json')
    tokenizer_path=Path(tokenizer_entry['path'])/'tokenizer.json'
    if data_inputs is not None:
        shared=read(data_inputs)['shared_tokenizer']
        if shared['sha256']!=tf['sha256']:raise ValueError('Shared tokenizer differs')
        tokenizer_path=Path(shared['local'])
    verify_file(tokenizer_path,tf)
    tok=Tokenizer.from_file(str(tokenizer_path));eos=tokenizer_entry['config']['eos_token_id']
    if not isinstance(eos,int):raise ValueError('Ambiguous document EOS')
    output.mkdir(parents=True,exist_ok=False)
    pool=np.memmap(old/'documents.bin',dtype=INDEX,mode='r')
    orders={2030:np.memmap(old/'document_order.bin',dtype='<u8',mode='r'),2031:np.random.default_rng(2031).permutation(len(pool)).astype('<u8')}
    if len(orders[2030])!=len(pool):raise ValueError('Pool/permutation length mismatch')
    targets=TARGETS
    lengths={s:candidate_count(pool,order,int(targets[s]*1.5)) for s,order in orders.items()}
    for s,order in orders.items():order.tofile(output/f'document_order_{s}.bin')
    offsets=np.memmap(output/'token_offsets.bin',dtype='<i8',mode='w+',shape=(len(pool),2));offsets[:]=-1
    selected=np.zeros(len(pool),dtype=bool);encoded=0;token_count=0;verified_raw={};started=time.monotonic();last=started
    with (output/'selected_tokens.bin').open('xb') as storage:
        for attempt in range(20):
            for s,order in orders.items():selected[order[:lengths[s]]]=True
            pending=selected & (offsets[:,0]<0)
            for source in np.unique(pool['source'][pending]):
                indices=np.flatnonzero(pending & (pool['source']==source))
                indices=indices[np.argsort(pool['row'][indices],kind='stable')];wanted=pool['row'][indices]
                record=plan['corpora']['fineweb_edu']['files'][int(source)]
                path=Path(record['local']);path=path/Path(record['path']).name if path.is_dir() else path
                if int(source) not in verified_raw:
                    verify_file(path,record);verified_raw[int(source)]=dict(path=str(path),sha256=record['sha256'],bytes=record['bytes'])
                base=0;matched=0
                for batch in pq.ParquetFile(path).iter_batches(batch_size=4096,columns=['text']):
                    lo=int(np.searchsorted(wanted,base));hi=int(np.searchsorted(wanted,base+len(batch)))
                    if hi>lo:
                        relative=(wanted[lo:hi]-base).astype('int64');texts=batch.column(0).take(relative).to_pylist()
                        for first in range(0,len(texts),64):
                            selected_text=texts[first:first+64];selected_ids=indices[lo+first:lo+first+len(selected_text)]
                            for text,index in zip(selected_text,selected_ids,strict=True):
                                if hashlib.sha256(text.encode('utf8')).digest()!=bytes(pool[int(index)]['hash']):
                                    raise ValueError('Raw document does not match historical pool hash')
                            encoded_text=tok.encode_batch(selected_text,add_special_tokens=False)
                            for ids,index in zip(encoded_text,selected_ids,strict=True):
                                arr=np.asarray(ids.ids+[eos],dtype='<u4');storage.write(arr.tobytes())
                                offsets[int(index)]=[token_count,len(arr)];token_count+=len(arr);encoded+=1;matched+=1
                            if time.monotonic()-last>=30:
                                write(output/'progress.json',dict(stage='retokenizing',round=attempt,documents=encoded,tokens=token_count,source=int(source),seconds=time.monotonic()-started))
                                last=time.monotonic()
                    base+=len(batch)
                    if hi==len(wanted):break
                if matched!=len(indices):raise ValueError('Raw shard is missing selected rows')
                storage.flush();offsets.flush()
                write(output/'progress.json',dict(stage='retokenizing',round=attempt,documents=encoded,tokens=token_count,source=int(source),seconds=time.monotonic()-started))
            enough={s:int(offsets[order[:lengths[s]],1].sum())>=targets[s] for s,order in orders.items()}
            if all(enough.values()):break
            for s,ok in enough.items():
                if not ok:
                    if lengths[s]==len(pool):raise ValueError('Entire historical pool too short after retokenization')
                    lengths[s]=min(len(pool),lengths[s]*2)
        else:raise ValueError('Cannot materialize required prefix after bounded expansion')
    result={}
    for s,order in orders.items():
        dest=output/str(s);info=emit_prefix(pool,order,offsets,output/'selected_tokens.bin',dest,targets[s],eos)
        value=dict(status='complete',schema='a2-family-retokenized-prefix-v1',streams={'heal':info},
                   seed=s,eos_id=eos,tokenizer_sha256=tf['sha256'],pool_documents_sha256=POOL_SHA,
                   document_count=len(pool),document_order_path=str(output/f'document_order_{s}.bin'),
                   document_order_sha256=sha(output/f'document_order_{s}.bin'),source_plan_sha256=PLAN_SHA,
                   source_manifest_sha256=MANIFEST_SHA,packing='continuous4096; explicit EOS; boundaries retained; no BOS',
                   text_hash_verified_documents=encoded,raw_files=verified_raw,prepared_prefix_tokens=targets[s],
                   attention_contract='intra-document mask plus reset positions, validated by worker FT2')
        lock=write(dest/'manifest.json',value);result[str(s)]=dict(path=str(dest),sha256=lock['sha256'])
    write(output/'complete.json',dict(status='complete',streams=result,elapsed_seconds=time.monotonic()-started))
    print('FAMILY_FWE_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('plan','old-shuffle','downloads','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--data-inputs')
    a=p.parse_args();prepare(a.plan,a.old_shuffle,a.downloads,a.output,a.data_inputs)
