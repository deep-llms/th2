"""Document shuffle of the completed 30B DCLM stream; no retokenization or GPU use."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scripts.a2_prepare_corpora import sha,verify_file,write_json,verify_packed

DOC=np.dtype([('row','<u8'),('hash','V32'),('tokens','<u8'),('group','u1')])
INDEX=np.dtype([('source','<u4'),('row','<u8'),('hash','V32'),('offset','<u8'),('tokens','<u8')])


def documents(root,plan,manifest,group,corpus="dclm"):
    """Recover ids, exact-text hashes, and offsets from pinned completed shards."""
    rows=[];offset=0;partial=0
    for item in manifest['source_prefixes']:
        if item['group']!=group:continue
        i=item['source_index'];part=root/'parts'/corpus/f'{i:05d}'
        receipt=part/'complete.json'
        if sha(receipt)!=item['shard_receipt_sha256']:raise ValueError('Source receipt changed')
        saved=json.loads(receipt.read_text())
        if saved['source']!=plan['corpora'][corpus]['files'][i]:raise ValueError('Source identity changed')
        verify_file(part/'documents.bin',saved['files']['documents.bin'])
        docs=np.fromfile(part/'documents.bin',dtype=DOC)
        docs=docs[docs['group']=={'heal':0,'tuning':1,'final':2}[group]]
        ends=np.cumsum(docs['tokens'],dtype=np.uint64);take=item['tokens']
        # Heal uses complete documents only. The original last partial document
        # is excluded before shuffle, not silently treated as a complete one.
        mask=ends<=take if group=='heal' else ends-docs['tokens']<take
        chosen=docs[mask];arr=np.empty(len(chosen),dtype=INDEX)
        arr['source']=i;arr['row']=chosen['row'];arr['hash']=chosen['hash'];arr['tokens']=chosen['tokens']
        arr['offset']=offset+(ends-docs['tokens'])[mask]
        rows.append(arr);offset+=take
        if group=='heal':partial+=take-int(chosen['tokens'].sum())
    return np.concatenate(rows),partial


def prepare(plan_path,source_sha,output,spec,length=4096,minimum=781248,*,corpus="dclm",prior_verification=None,replication_spec=None):
    started=time.monotonic();plan=json.loads(Path(plan_path).read_text());root=Path(plan['root']);source=root/'packed'/corpus
    if sha(source/'manifest.json')!=source_sha:raise ValueError('Unreviewed corpus manifest')
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest['plan_sha256']!=sha(plan_path):raise ValueError('Different source plan')
    if corpus not in ('dclm','fineweb_edu'):raise ValueError('Unregistered corpus')
    if prior_verification is None:
        verify_packed(source,manifest)
    else:
        receipt=json.loads(Path(prior_verification).read_text())
        if (receipt.get('status')!='verified' or receipt.get('corpus')!=corpus
            or receipt.get('manifest_sha256')!=source_sha
            or not receipt.get('full_output_hashes_passed') or not receipt.get('native_reader_passed')
            or receipt['streams']!=manifest['streams']):raise ValueError('Prior full verification differs')
        for name,stream in manifest['streams'].items():
            for suffix,mult in [('bin',4),('starts',1)]:
                if (source/(name+'.'+suffix)).stat().st_size!=stream['tokens']*mult:raise ValueError('Source size changed')

    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    heal,excluded=documents(root,plan,manifest,'heal',corpus);held,_=documents(root,plan,manifest,'tuning',corpus)
    ids=lambda a:np.rec.fromarrays([a['source'],a['row']],names='source,row')
    if np.intersect1d(ids(heal),ids(held)).size or np.intersect1d(heal['hash'],held['hash']).size:
        raise ValueError('T7 overlap by document id or exact content')
    total=int(heal['tokens'].sum());full=total//length*length
    if full//length<minimum:raise ValueError('Insufficient complete-document training data')
    order=np.random.default_rng(2030).permutation(len(heal))
    heal.tofile(output/'documents.bin');held.tofile(output/'heldout_documents.bin');order.astype('<u8').tofile(output/'document_order.bin')
    write_json(output/'status.json',dict(status='copying',documents=len(heal),tokens=full))
    source_ids=np.memmap(source/'heal.bin',dtype='<u4',mode='r');source_starts=np.memmap(source/'heal.starts',dtype='u1',mode='r')
    count=eos=0
    with (output/'heal.bin').open('xb',buffering=8<<20) as f,(output/'heal.starts').open('xb',buffering=8<<20) as g:
        for position,index in enumerate(order):
            doc=heal[index];start=int(doc['offset']);n=int(doc['tokens'])
            if source_starts[start]!=1 or source_ids[start+n-1]!=0:raise ValueError('Invalid original document boundary/EOS')
            take=min(n,full-count)
            if take:
                tokens=source_ids[start:start+take];f.write(tokens.tobytes());g.write(source_starts[start:start+take].tobytes())
                eos+=int(np.count_nonzero(tokens==0));count+=take
            if position%100000==0:print('SHUFFLE_PROGRESS',position,count,flush=True)
            if count==full:break
    info=dict(rows=full//length,tokens=full,length=length,token_sha256=sha(output/'heal.bin'),starts_sha256=sha(output/'heal.starts'))
    result=dict(status='complete',schema='a2-stage1-shuffled-v1',spec_sha256=sha(spec),source_manifest_sha256=source_sha,
        source_plan_sha256=sha(plan_path),tokenizer_sha256=manifest['tokenizer_sha256'],eos_id=0,streams={'heal':info},
        order='numpy.random.default_rng(2030).permutation over complete documents; sequential packed rows in every run',
        excluded_original_tail_tokens=excluded,dropped_after_shuffle=total-full,document_count=len(heal),
        documents_sha256=sha(output/'documents.bin'),document_order_sha256=sha(output/'document_order.bin'),
        heldout_documents_sha256=sha(output/'heldout_documents.bin'),document_dtype=INDEX.descr,
        heldout=dict(path=str(source),stream='tuning',manifest_sha256=source_sha,tokens=manifest['streams']['tuning']['tokens']),
        T7=dict(passed=True,id_overlap=0,content_hash_overlap=0),
        T18=dict(sequences=full//length,eos_tokens=eos,eos_share=eos/full),elapsed_seconds=time.monotonic()-started)
    if replication_spec:
        result['stage1b_spec_sha256']=sha(replication_spec)
        result['corpus']=corpus
        result['prior_verification_sha256']=sha(prior_verification) if prior_verification else None
    verify_packed(output,result);write_json(output/'manifest.json',result)
    from efficient_llm.a2.data import PackedRows
    reader=PackedRows(output,'heal',count=2,order=[0,info['rows']-1],verify=False)
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(Path(plan['tokenizer_path'])/'tokenizer.json'))
    write_json(output/'diagnostics.json',dict(first_sequence=tok.decode(reader[0]['input_ids'].tolist(),skip_special_tokens=False),last_sequence=tok.decode(reader[1]['input_ids'].tolist(),skip_special_tokens=False)))
    write_json(output/'complete.json',dict(status='complete',manifest_sha256=sha(output/'manifest.json'),native_reader_passed=True,seconds=time.monotonic()-started))
    print('STAGE1_DATA_COMPLETE',json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser()
    for k in ('plan','output','spec'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--source-sha256',required=True);a=p.parse_args()
    prepare(a.plan,a.source_sha256,a.output,a.spec)
if __name__=='__main__':main()
