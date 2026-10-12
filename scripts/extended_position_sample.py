"""Materialize a frozen, overlap-audited selection from local original documents."""
import hashlib
import json
from pathlib import Path

from eval.paired import position, protocol
from eval.paired.run import token_hash


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def materialize(selection_path,old_rows,new_root,old_manifest,tokenizer,output,frozen):
    from datasets import Dataset
    selection=json.loads(Path(selection_path).read_text())
    audit_path=(Path(selection_path).parent/selection['overlap_audit_file']).resolve()
    if not audit_path.is_relative_to(Path(selection_path).resolve().parent) or sha(audit_path)!=selection['overlap_audit_sha256']:
        raise ValueError('Position overlap audit differs')
    audit=json.loads(audit_path.read_text())
    if audit['old_manifest_sha256']!=selection['old_manifest_sha256']:
        raise ValueError('Overlap audit refers to a different training manifest')
    excluded=set(audit['excluded_text_sha256'])
    if sha(old_manifest)!=selection['old_manifest_sha256']:
        raise ValueError('Original training manifest differs from overlap audit')
    manifest=json.loads(Path(old_manifest).read_text())
    expected={p:f for p,f in manifest['files'].items() if p.startswith('subsets/qwen3_0.6b_base_en_30B/train/') and p.endswith('.arrow')}
    checked={f['path']:{k:f[k] for k in ('bytes','sha256')} for f in audit['files']}
    expected={p:{k:f[k] for k in ('bytes','sha256')} for p,f in expected.items()}
    if not expected or checked!=expected or len(checked)!=len(audit['files']):
        raise ValueError('Overlap audit does not cover every original training file')
    if len(selection['rows'])!=frozen['position']['documents']:
        raise ValueError('Position selection count differs from protocol')
    if token_hash(tokenizer)!=selection['tokenizer_sha256']:
        raise ValueError('Position selection tokenizer differs')
    sources={'old_validation':old_rows};root=Path(new_root).resolve()
    for name,source in selection['sources'].items():
        if name=='old_validation':raise ValueError('Cannot replace original validation source')
        path=(root/source['path']).resolve()
        if not path.is_relative_to(root) or path.stat().st_size!=source['bytes'] or sha(path)!=source['sha256']:
            raise ValueError('Extended position source differs: '+name)
        sources[name]=Dataset.from_file(str(path))
    rows=[];seen=set();prefixes=set();length=frozen['position']['length']
    for item in selection['rows']:
        if type(item['row']) is not int or item['row']<0:raise ValueError('Invalid source row')
        text=sources[item['source']][item['row']]['text']
        digest=hashlib.sha256(text.encode()).hexdigest()
        if digest in excluded:raise ValueError('Position document overlaps training')
        if digest!=item['text_sha256'] or digest in seen:
            raise ValueError('Position text differs or is duplicated')
        ids=tokenizer(text,add_special_tokens=False)['input_ids'][:length]
        prefix=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        if len(ids)!=length or prefix!=item['prefix_sha256'] or prefix in prefixes:
            raise ValueError('Position prefix differs, is short or duplicated')
        seen.add(digest);prefixes.add(prefix)
        rows.append(dict(document_id=f"{item['source']}:{item['row']}:{digest}",
                         token_sha256=protocol.digest(ids),input_ids=ids))
    sample=dict(protocol_sha256=frozen['protocol_sha256'],tokenizer_sha256=token_hash(tokenizer),
        policy='Frozen source/row order; exact full-text overlap excluded against entire old English training pool; no near-duplicate claim; tokenize without special tokens; truncate; no appended EOS',
        selection_sha256=sha(selection_path),old_training_manifest_sha256=selection['old_manifest_sha256'],
        overlap_audit_sha256=selection['overlap_audit_sha256'],length=length,rows=rows)
    sample['sample_sha256']=protocol.digest(sample)
    protocol.write_json(output,sample)
    return position.read_sample(output,frozen)
