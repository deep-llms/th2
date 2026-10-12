"""CPU-only deterministic selection; no model scoring or network access.

Reads local published old/new CulturaX repositories. Excludes exact full-text
matches against ALL old English training documents, including those beyond 20k.
The newer pool's train split is eligible only because it did not train these
checkpoints, and only after that explicit exclusion. No near-duplicate claim.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.extended_position_sample import sha

SUBSET='subsets/qwen3_0.6b_base_en_30B'


def candidates(old,new,tokenizer):
    from datasets import load_from_disk
    rows=[];seen=set();prefixes=set()
    for name,root,split in [('old_validation',old,'validation'),('new_validation',new,'validation'),
                            ('new_train_unused_candidates',new,'train')]:
        ds=load_from_disk(root/SUBSET/split/'shard_000000');count=0
        for start in range(0,len(ds),128):
            texts=ds[start:start+128]['text'];sequences=tokenizer(texts,add_special_tokens=False)['input_ids']
            for offset,(text,ids) in enumerate(zip(texts,sequences)):
                if len(ids)<2048:continue
                digest=hashlib.sha256(text.encode()).hexdigest()
                prefix=hashlib.sha256(json.dumps(ids[:2048]).encode()).hexdigest()
                if digest in seen or prefix in prefixes:continue
                seen.add(digest);prefixes.add(prefix);count+=1
                rows.append(dict(source=name,row=start+offset,text_sha256=digest,prefix_sha256=prefix,text=text))
            if name=='new_train_unused_candidates' and count>=1800:break
        print('CANDIDATES',name,count,flush=True)
    return rows


def overlap(old,rows):
    import pyarrow as pa
    import pyarrow.compute as pc
    manifest=old/'dataset_manifest.json';files=json.loads(manifest.read_text())['files']
    values=pa.array([r['text'] for r in rows]);matches=set();checked=[];count=0
    for rel,spec in files.items():
        if not rel.startswith(SUBSET+'/train/') or not rel.endswith('.arrow'):continue
        with pa.memory_map(str(old/rel),'r') as source:
            buf=source.read_buffer();digest=hashlib.sha256(buf).hexdigest()
            if len(buf)!=spec['bytes'] or digest!=spec['sha256']:raise ValueError('Training file differs: '+rel)
            text=pa.ipc.open_stream(pa.BufferReader(buf)).read_all()['text'];count+=len(text)
            for value in pc.filter(text,pc.is_in(text,value_set=values)).to_pylist():
                matches.add(hashlib.sha256(value.encode()).hexdigest())
        checked.append(dict(path=rel,sha256=digest,bytes=spec['bytes']))
        print('OVERLAP',len(checked),count,len(matches),flush=True)
    if not checked:raise ValueError('No original training files scanned')
    return dict(method='Exact UTF-8 text identity against the complete original English training pool; no near-duplicate claim',
        old_manifest_sha256=sha(manifest),files=checked,training_documents=count,
        excluded_text_sha256=sorted(matches),candidates=len(values))


def publish(rows,audit,new,tokenizer,sources,selection_path,audit_path):
    from eval.paired.run import token_hash
    from eval.paired.protocol import write_json
    excluded=set(audit['excluded_text_sha256'])
    selected=[{k:v for k,v in r.items() if k!='text'} for r in rows if r['text_sha256'] not in excluded][:2000]
    if len(selected)!=2000:raise ValueError('Insufficient non-overlapping unique documents')
    required={r['source'] for r in selected}-{'old_validation'}
    if not required<=set(sources):raise ValueError('Missing named extension sources')
    for source in sources.values():
        if sha(new/source['path'])!=source['sha256']:raise ValueError('New source changed')
    write_json(audit_path,audit)
    value=dict(documents=2000,length=2048,old_manifest_sha256=audit['old_manifest_sha256'],
        tokenizer_sha256=token_hash(tokenizer),overlap_audit_sha256=sha(audit_path),
        overlap_audit_file=Path(audit_path).name,sources=sources,rows=selected,
        counts=dict(Counter(r['source'] for r in selected)),
        selection_policy='Old validation, then new validation, then new pool train shard 0; stored row order; unique text and token prefix; exclude all old-training exact text matches before first 2000')
    write_json(selection_path,value)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('old-root','new-root','sources','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.old_root/'provenance/tokenizer',local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    rows=candidates(a.old_root,a.new_root,tok);audit=overlap(a.old_root,rows)
    publish(rows,audit,a.new_root,tok,json.loads(a.sources.read_text()),a.output/'selection.json',a.output/'overlap.json')
