"""Assemble hash-exact recovered/raw CX documents, then reproduce A2 S tokenization."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.a2_prepare_corpora import verify_file
from efficient_llm.a2_family.common import read, write, sha, SPEC_SHA, INPUT_PREPARATION_SPEC_SHA


def prepare(inputs, output, reference):
    import pyarrow as pa
    from tokenizers import Tokenizer
    cfg=read(inputs);output=Path(output)
    if cfg['spec_sha256'] not in (SPEC_SHA,INPUT_PREPARATION_SPEC_SHA):raise ValueError('Wrong Family inputs')
    recovery=Path(cfg['recovery_root']);raw_text=recovery/'exact_documents.jsonl'
    if sha(raw_text)!=cfg['recovery_document_sha256']:raise ValueError('Recovered text artifact changed')
    receipt=read(recovery/'recovery.json');docs={g:{} for g in ('tuning','final')}
    with raw_text.open() as f:
        for line in f:
            d=json.loads(line)
            if hashlib.sha256(d['text'].encode()).hexdigest()!=d['text_sha256']:raise ValueError('Recovered text hash differs')
            docs[d['group']][d['sequence_index']]=d
    raw_sources={}
    for source in cfg['files']:
        path=Path(source['local']);verify_file(path,source)
        with pa.memory_map(str(path),'r') as stream:
            table=pa.ipc.open_stream(stream).read_all()
        raw_sources[source['name']]=table
    for item in cfg['missing']:
        text=raw_sources[item['source']]['text'][item['local_row']].as_py()
        if hashlib.sha256(text.encode()).hexdigest()!=item['text_sha256']:raise ValueError('Raw fallback text hash differs')
        if item['sequence_index'] in docs[item['group']]:raise ValueError('Duplicate document')
        docs[item['group']][item['sequence_index']]=dict(item,text=text)
    tok_file=cfg['shared_tokenizer'];verify_file(tok_file['local'],tok_file)
    tok=Tokenizer.from_file(tok_file['local']);eos=cfg['tokenizer_eos_id']
    output.mkdir(parents=True,exist_ok=False);sets={}
    for group,table in docs.items():
        if sorted(table)!=list(range(receipt['counts'][group])):raise ValueError('Incomplete or reordered evaluation set')
        dest=output/group;dest.mkdir();tokens=[];offsets=[0];metadata=[]
        for i in range(len(table)):
            d=table[i];ids=(tok.encode(d['text'],add_special_tokens=False).ids+[eos])[:4096]
            tokens.extend(ids);offsets.append(len(tokens))
            metadata.append(dict(index=i,source_split=d['split'],source_index=d['index'],text_sha256=d['text_sha256'],tokens=len(ids),predicted_tokens=max(len(ids)-1,0)))
        np.save(dest/'tokens.npy',np.asarray(tokens,dtype=np.int32));np.save(dest/'offsets.npy',np.asarray(offsets,dtype=np.int64))
        write(dest/'documents.json',dict(documents=metadata))
        info=dict(documents=len(metadata),tokens=len(tokens),predicted_tokens=sum(d['predicted_tokens'] for d in metadata),
                  token_sha256=sha(dest/'tokens.npy'),offset_sha256=sha(dest/'offsets.npy'),documents_sha256=sha(dest/'documents.json'),
                  tokenizer_sha256=tok_file['sha256'],convention='no BOS/template, explicit EOS then truncate4096; one document per sequence')
        write(dest/'manifest.json',info);sets[group]=info
    old=read(reference)
    checks=[dict(test='A2S_final_token_byte_identity',passed=all(sets['final'][k]==old[k] for k in ('documents','tokens','predicted_tokens','token_sha256','offset_sha256')),expected=old,actual=sets['final'])]
    write(output/'complete.json',dict(status='complete',sets=sets,checks=checks,unverified=not all(c['passed'] for c in checks),input_sha256=sha(inputs),exact_text_hashes=receipt['counts']))
    print('FAMILY_CX_COMPLETE',json.dumps(dict(sets=sets,checks=checks)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ('inputs','output','reference'):p.add_argument('--'+x,required=True)
    a=p.parse_args();prepare(a.inputs,a.output,a.reference)
