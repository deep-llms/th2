"""Seeded full-pool document permutations, materializing only registered Y1 prefix."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.a2_stage1_data import INDEX, documents
from scripts.a2_prepare_corpora import sha, verify_packed
from efficient_llm.a2_why.common import read,write,Checks,SPEC_SHA


def prefix_copy(heal,order,source,output,full):
    ids=np.memmap(source/'heal.bin',dtype='<u4',mode='r');starts=np.memmap(source/'heal.starts',dtype='u1',mode='r')
    count=eos=0
    with (output/'heal.bin').open('xb',buffering=8<<20) as f,(output/'heal.starts').open('xb',buffering=8<<20) as g:
        for index in order:
            doc=heal[index];start=int(doc['offset']);n=int(doc['tokens'])
            if starts[start]!=1 or ids[start+n-1]!=0:raise ValueError('Invalid source boundary/EOS')
            take=min(n,full-count);tokens=ids[start:start+take]
            f.write(tokens.tobytes());g.write(starts[start:start+take].tobytes())
            count+=take;eos+=int(np.count_nonzero(tokens==0))
            if count==full:break
    if count!=full:raise ValueError('Prefix shortfall')
    return eos


def prepare(plan_path, corpus, old_shuffle, output, cx_data):
    plan=read(plan_path);source=Path(plan['root'])/'packed'/corpus;manifest=read(source/'manifest.json')
    verify_packed(source,manifest)
    old=read(Path(old_shuffle)/'manifest.json');output=Path(output);output.mkdir(parents=True,exist_ok=False)
    checks=Checks(output/'tests.json')
    heal,excluded=documents(Path(plan['root']),plan,manifest,'heal',corpus)
    checks.record('historical_document_pool',sha(Path(old_shuffle)/'documents.bin')==old['documents_sha256'] and
                  len(heal)==old['document_count'] and np.array_equal(heal,np.fromfile(Path(old_shuffle)/'documents.bin',dtype=INDEX)))
    held={name:documents(Path(plan['root']),plan,manifest,name,corpus)[0] for name in ('tuning','final')}
    ids=lambda x:np.rec.fromarrays([x['source'],x['row']],names='source,row')
    # Verify whole source pool, not only the selected prefix.
    for name,arr in held.items():
        overlap_id=int(np.intersect1d(ids(heal),ids(arr)).size)
        overlap_hash=int(np.intersect1d(heal['hash'],arr['hash']).size)
        checks.record('YT2_corpus_'+name,overlap_id==0 and overlap_hash==0,dict(id_overlap=overlap_id,content_hash_overlap=overlap_hash))
    cx_manifest=read(Path(cx_data)/'manifest.json')
    checks.record('CulturaX_document_manifest',sha(Path(cx_data)/'documents.jsonl')==cx_manifest['documents_sha256'])
    hashes=[]
    with (Path(cx_data)/'documents.jsonl').open() as f:
        for line in f:
            record=json.loads(line)
            if record.get('group') in ('tuning','final') or record.get('split')=='validation':hashes.append(bytes.fromhex(record['text_sha256']))
    overlap=int(np.intersect1d(heal['hash'],np.asarray(hashes,dtype='V32')).size)
    checks.record('YT2_CulturaX_heldout',len(hashes)>0 and overlap==0,
        dict(content_hash_overlap=overlap,heldout_documents=len(hashes),id_disjoint='Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index'))
    result={}
    for seed in (2031,2032):
        dest=output/str(seed);dest.mkdir();order=np.random.default_rng(seed).permutation(len(heal))
        order.astype('<u8').tofile(dest/'document_order.bin')
        checks.record('YT2_new_permutation_'+str(seed),sha(dest/'document_order.bin')!=old['document_order_sha256'])
        full=762*64*4096;eos=prefix_copy(heal,order,source,dest,full)
        info=dict(rows=762*64,tokens=full,length=4096,token_sha256=sha(dest/'heal.bin'),starts_sha256=sha(dest/'heal.starts'))
        oldids=np.memmap(Path(old_shuffle)/'heal.bin',dtype='<u4',mode='r',shape=(full,))
        import hashlib
        checks.record('YT2_different_tokens_'+str(seed),info['token_sha256']!=hashlib.sha256(oldids).hexdigest())
        value=dict(status='complete',schema='a2-why-shuffled-prefix-v1',source_manifest_sha256=sha(source/'manifest.json'),
            source_plan_sha256=sha(plan_path),tokenizer_sha256=manifest['tokenizer_sha256'],eos_id=0,streams={'heal':info},
            seed=seed,document_count=len(heal),document_order_sha256=sha(dest/'document_order.bin'),
            excluded_original_tail_tokens=excluded,pool_documents_sha256=old['documents_sha256'],
            order=f'numpy.random.default_rng({seed}).permutation over entire complete-document pool; sequential packed rows',
            materialization='Exact first 762 updates of full shuffled stream; no sampling restriction before permutation',
            eos_tokens=eos,checks=checks.items,unverified=any(not c['passed'] for c in checks.items))
        verify_packed(dest,value);lock=write(dest/'manifest.json',value)
        result[str(seed)]=dict(path=str(dest),sha256=lock['sha256'])
    write(output/'complete.json',dict(status='complete',streams=result,checks=checks.items))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ('plan','corpus','old-shuffle','output','cx-data'):p.add_argument('--'+x,required=True)
    a=p.parse_args();prepare(a.plan,a.corpus,a.old_shuffle,a.output,a.cx_data)
