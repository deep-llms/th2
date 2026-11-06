"""Recover exact CulturaX text from existing byte-level tokens, hash every document.

A decoded document is accepted ONLY when its UTF-8 SHA equals the original
recorded text SHA. Missing tails or any mismatch require the original HF shard.
This is an offline CPU input preparation job; it never signals a GPU process.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from efficient_llm.a2_family.common import write, read, sha, SPEC_SHA

CX_MANIFEST='35aa415f1a3a2a60d9d3faa7faa85e97ce365088f38d23ea01b4a18737834c3e'
CX_DOCUMENTS='5d5b3b53445bd9d3939f9148de8800c237cdf393852634ba6c9bdc11c1a16ad5'
TOKENIZER='c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624'

def recover(data, tokenizer_file, output):
    data,output=Path(data),Path(output)
    if sha(data/'manifest.json')!=CX_MANIFEST or sha(data/'documents.jsonl')!=CX_DOCUMENTS:
        raise ValueError('CulturaX provenance identity changed')
    if sha(tokenizer_file)!=TOKENIZER:raise ValueError('GPT-NeoX tokenizer identity changed')
    manifest=read(data/'manifest.json')
    arrays={}
    for group in ('tuning','final'):
        desc=manifest['streams'][group]
        if sha(data/(group+'.bin'))!=desc['token_sha256']:raise ValueError('Packed tokens differ')
        arrays[group]=np.memmap(data/(group+'.bin'),dtype='<u4',mode='r')
    tok=Tokenizer.from_file(str(tokenizer_file))
    output.mkdir(parents=True,exist_ok=False)
    offsets=dict(tuning=0,final=0);missing=[];counts=dict(tuning=0,final=0);recovered=dict(tuning=0,final=0)
    with (output/'exact_documents.jsonl').open('x') as out,(data/'documents.jsonl').open() as src:
        for line in src:
            d=json.loads(line);group=d['group']
            if group not in arrays:continue
            ordinal=counts[group];counts[group]+=1
            start=offsets[group];length=d['gpt_neox_tokens'];offsets[group]+=length
            info=dict(d,sequence_index=ordinal)
            reason=None;text=None
            if start+length>len(arrays[group]):reason='Original packing clipped the document tail'
            elif int(arrays[group][start+length-1])!=0:reason='Expected appended EOS absent'
            else:
                text=tok.decode(arrays[group][start:start+length-1].tolist(),skip_special_tokens=False)
                if hashlib.sha256(text.encode('utf8')).hexdigest()!=d['text_sha256']:
                    reason='Decoded UTF-8 differs from original text hash'
            if reason:
                missing.append(dict(info,reason=reason))
            else:
                out.write(json.dumps(dict(info,text=text),ensure_ascii=False)+'\n');recovered[group]+=1
    result=dict(status='complete' if not missing else 'awaiting_exact_raw_documents',counts=counts,recovered=recovered,
                missing=missing,documents_sha256=sha(output/'exact_documents.jsonl'),
                source_manifest_sha256=CX_MANIFEST,source_documents_sha256=CX_DOCUMENTS,
                decoded_text_sha_checks_passed=sum(recovered.values()),
                method='Lossless text recovery proved independently by every original SHA; no approximation accepted')
    write(output/'recovery.json',result)
    print('FAMILY_CX_RECOVERY',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('data','tokenizer-file','output'):p.add_argument('--'+field,required=True)
    a=p.parse_args();recover(a.data,a.tokenizer_file,a.output)
