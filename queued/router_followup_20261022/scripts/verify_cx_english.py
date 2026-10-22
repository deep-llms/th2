"""Verify controller-downloaded English pool and tokenizer before training, offline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--contract',type=Path,default=Path(__file__).resolve().parents[1]/'resources/cx_english_20261020.json')
    a=p.parse_args();contract=json.loads(a.contract.read_text())
    if a.output.exists():raise ValueError('Existing verification receipt')
    subset=a.data_root/'subsets'/contract['subset']
    for file,expected in [(subset/'manifest.json',contract['manifest_sha256']),
                          (a.data_root/'UPLOAD_COMPLETE.json',contract['completion_sha256'])]:
        if sha256(file)!=expected:raise ValueError(f'Metadata checksum mismatch: {file}')
    manifest=json.loads((subset/'manifest.json').read_text())
    marker=json.loads((a.data_root/'UPLOAD_COMPLETE.json').read_text())
    if marker['state']!='complete' or manifest['totals']!=contract['totals']:raise ValueError('Pool metadata mismatch')
    files={}
    for split in ('train','validation'):
        expected_dirs={Path(s['path']).name for s in manifest['shards'][split]}
        actual_dirs={s.name for s in (subset/split).iterdir() if s.is_dir()}
        if actual_dirs!=expected_dirs:raise ValueError(f'Unexpected {split} shard directories')
        for shard in manifest['shards'][split]:files.update(shard['files'])
    if len(files)!=contract['files']:raise ValueError('Unexpected file count')
    for file,meta in files.items():
        path=(a.data_root/file).resolve()
        if not path.is_relative_to(a.data_root.resolve()):raise ValueError('Unsafe manifest path')
        if path.stat().st_size!=meta['bytes']:raise ValueError(f'Incomplete file: {file}')
    def check(item):
        file,meta=item
        if sha256(a.data_root/file)!=meta['sha256']:raise ValueError(f'Checksum mismatch: {file}')
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,_ in enumerate(pool.map(check,files.items()),1):
            if i%250==0:print(f'POOL_HASH_PROGRESS {i}/{len(files)}',flush=True)
    for name,meta in contract['tokenizer_files'].items():
        file=a.model_root/name
        if file.stat().st_size!=meta['bytes'] or sha256(file)!=meta['sha256']:
            raise ValueError(f'Tokenizer/config checksum mismatch: {name}')
    from transformers import AutoTokenizer, AutoConfig
    tokenizer=AutoTokenizer.from_pretrained(a.model_root,local_files_only=True)
    if tokenizer.encode('<|endoftext|>',add_special_tokens=False)!=[151643]:raise ValueError('Wrong EOD token')
    cfg=AutoConfig.from_pretrained(a.model_root,local_files_only=True)
    expected=dict(model_type='qwen3',num_hidden_layers=28,hidden_size=1024,intermediate_size=3072,
                  num_attention_heads=16,num_key_value_heads=8,head_dim=128,vocab_size=151936,tie_word_embeddings=True)
    if any(getattr(cfg,k)!=v for k,v in expected.items()):raise ValueError('Unexpected original model configuration')
    # Only complete save_to_disk shards are accepted, including their metadata.
    from training_data import load_text
    split_results={}
    for split in ('train','validation'):
        ds,sources=load_text(subset/split)
        if len(ds)!=manifest['totals'][split]['documents']:raise ValueError('Wrong document count')
        split_results[split]=dict(documents=len(ds),shards=len(sources),fingerprint=ds._fingerprint)
    receipt=dict(status='passed',contract=contract,files_verified=len(files),splits=split_results,
                 data_root=str(a.data_root),subset_root=str(subset),tokenizer=str(a.model_root),
                 tokenizer_files={p.name:sha256(p) for p in sorted(a.model_root.iterdir()) if p.is_file()},
                 original_config=cfg.to_dict())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(receipt,f,indent=2)
    print('CX_ENGLISH_VERIFIED '+json.dumps(dict(status='passed',files=len(files),splits=split_results)),flush=True)


if __name__=='__main__':main()
