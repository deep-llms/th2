"""A2 document sampling and continuous EOS packing; all inputs are local."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from efficient_llm.diagnostic_data import sha256


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class PackedWriter:
    """Continuous packing across ALL documents, not separately in map batches."""
    def __init__(self, root, name, length, max_rows=None):
        self.root, self.name, self.length, self.max_rows = Path(root), name, length, max_rows
        self.tokens = (self.root/f'{name}.bin').open('xb')
        self.starts = (self.root/f'{name}.starts').open('xb')
        self.count = 0

    def add(self, ids):
        if not ids: raise ValueError('Documents must include EOS')
        keep = len(ids) if self.max_rows is None else min(len(ids), self.max_rows*self.length-self.count)
        if keep <= 0: return
        arr = np.asarray(ids[:keep], dtype='<u4')
        boundaries = np.zeros(keep, dtype='u1'); boundaries[0] = 1
        self.tokens.write(arr.tobytes()); self.starts.write(boundaries.tobytes()); self.count += keep

    def close(self):
        full = self.count // self.length * self.length
        self.tokens.truncate(full*4); self.starts.truncate(full)
        self.tokens.close(); self.starts.close()
        if full == 0: raise ValueError(f'No full sequences in {self.name}')
        return dict(rows=full//self.length, tokens=full, dropped=self.count-full, length=self.length,
                    token_sha256=sha256(self.root/f'{self.name}.bin'),
                    starts_sha256=sha256(self.root/f'{self.name}.starts'))


class PackedRows(Dataset):
    def __init__(self, root, name, *, count=None, order=None, verify=True):
        self.root = Path(root)
        manifest = json.loads((self.root/'manifest.json').read_text())
        info = manifest['streams'][name]
        if manifest['status'] != 'complete': raise ValueError('Incomplete data artifact')
        self.length = info['length']; self.total = info['rows']
        if verify and (sha256(self.root/f'{name}.bin') != info['token_sha256']
                       or sha256(self.root/f'{name}.starts') != info['starts_sha256']):
            raise ValueError('Packed data digest mismatch')
        self.ids = np.memmap(self.root/f'{name}.bin', mode='r', dtype='<u4', shape=(self.total,self.length))
        self.starts = np.memmap(self.root/f'{name}.starts', mode='r', dtype='u1', shape=(self.total,self.length))
        n = self.total if count is None else count
        if not 0 < n <= self.total: raise ValueError('Insufficient packed sequences')
        self.order = np.arange(n) if order is None else np.asarray(order)
        if len(self.order) != n or len(set(self.order.tolist())) != n or min(self.order) < 0 or max(self.order) >= self.total:
            raise ValueError('Invalid row selection/order')

    def __len__(self): return len(self.order)
    def __getitem__(self, index):
        i = int(self.order[index])
        return dict(input_ids=torch.from_numpy(self.ids[i].astype('int64')), sample_id=i)
    def boundaries(self, index):
        out = torch.from_numpy(self.starts[int(self.order[index])].astype(bool)); out[0] = True
        return out


def load_pool(root):
    """Read the published cx_sampled manifest order, including every shard."""
    from datasets import concatenate_datasets, load_from_disk
    root = Path(root)
    subset = root/'subsets'/'qwen3_0.6b_base_en_30B'
    manifest = json.loads((subset/'manifest.json').read_text())
    result = {}; files = {}; started = time.monotonic()
    for split in ('train','validation'):
        pieces = []
        for shard_number, shard in enumerate(manifest['shards'][split], 1):
            states = []
            for rel, expected in shard['files'].items():
                p = (root/rel).resolve()
                if not p.is_relative_to(root.resolve()): raise ValueError('Escaping shard path')
                if p.stat().st_size != expected['bytes'] or sha256(p) != expected['sha256']:
                    raise ValueError(f'Shard digest mismatch: {rel}')
                files[rel] = expected
                if p.name == 'state.json': states.append(p)
            if len(states) != 1: raise ValueError('Expected one saved dataset per shard')
            import pyarrow.parquet as pq
            ds = load_from_disk(str(states[0].parent))
            index_path = states[0].parent/'document_index.parquet'
            if str(index_path.relative_to(root.resolve())) not in files:
                raise ValueError('Token-count index was not checksum verified')
            metadata = pq.read_table(index_path, columns=['content_tokens','tokens_with_eod'])
            counts = metadata['tokens_with_eod'].to_numpy()
            content = metadata['content_tokens'].to_numpy()
            if len(counts) != len(ds) or not np.array_equal(counts, content+1) or (counts < 1).any():
                raise ValueError('Invalid published Qwen token counts')
            pieces.append(ds.add_column('a2_qwen_tokens_with_eod', counts))
            if shard_number % 100 == 0 or shard_number == len(manifest['shards'][split]):
                print('A2_POOL_VERIFIED', json.dumps(dict(split=split, shards=shard_number,
                    total_shards=len(manifest['shards'][split]), elapsed_seconds=time.monotonic()-started)), flush=True)
        result[split] = concatenate_datasets(pieces)
    return result, dict(manifest_sha256=sha256(subset/'manifest.json'), files=files)


def prepare(train, validation, qwen_tokenizer, tokenizer, output, *, provenance,
            length=4096, reserved_qwen=1_200_000_000, heal_tokens=850_000_000,
            tune_rows=1024, small_rows=256, batch_size=64):
    """Threshold arguments exist for CPU fixtures; production CLI fixes spec values.

    Qwen counts use text tokens plus one document EOS, recorded explicitly.
    No long-document filtering, no new shuffle, no truncation during tokenization.
    """
    if tokenizer.eos_token_id != 0 or min(length,reserved_qwen,heal_tokens,tune_rows,small_rows,batch_size) < 1:
        raise ValueError('Invalid tokenizer/budgets')
    if small_rows > tune_rows: raise ValueError('Small set exceeds tuning set')
    started = last_progress = time.monotonic()
    indexed_counts = 'a2_qwen_tokens_with_eod' in getattr(train, 'column_names', [])
    if indexed_counts:
        # Verify the convention on a fixed sample, then reuse the checksum-verified
        # published counts instead of re-tokenizing 1.2B tokens with Qwen.
        probe = np.random.default_rng(2028).permutation(len(train))[:64].tolist()
        texts = [train[i]['text'] for i in probe]
        encoded = qwen_tokenizer(texts, add_special_tokens=False, truncation=False)['input_ids']
        if any(len(ids)+1 != train[i]['a2_qwen_tokens_with_eod'] for i, ids in zip(probe, encoded, strict=True)):
            raise ValueError('Published token counts/tokenizer convention differ')
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    writers = {k:PackedWriter(root,k,length,tune_rows if k == 'tuning' else None)
               for k in ('heal','tuning','final')}
    counts = dict(heal=0,tuning=0,unused=0,final=0)
    sums = dict(heal=0,tuning=0,unused=0,final=0)
    seen = {}; qwen_sum = 0; extended = False; reserved_prefix = None
    order = np.random.default_rng(2028).permutation(len(train))
    ids_by_split = {k:[] for k in counts}
    def register(text, key):
        digest = hashlib.sha256(text.encode('utf8')).hexdigest()
        if key != 'unused':
            if digest in seen and seen[digest] != key:
                raise ValueError('T7 content overlap between '+seen[digest]+' and '+key)
            seen[digest] = key
        return digest
    # A failure deliberately leaves an incomplete directory without manifest.json.
    try:
        with (root/'documents.jsonl').open('x') as docs:
            finished = False
            for start in range(0,len(order),batch_size):
                indices = order[start:start+batch_size].tolist()
                texts = [train[i]['text'] for i in indices]
                qw = ([int(train[i]['a2_qwen_tokens_with_eod']) for i in indices] if indexed_counts else
                      [len(ids)+1 for ids in qwen_tokenizer(texts, add_special_tokens=False, truncation=False)['input_ids']])
                neo = tokenizer(texts, add_special_tokens=False, truncation=False)['input_ids']
                for index,text,q,n in zip(indices,texts,qw,neo,strict=True):
                    tokens = list(n)+[0]  # explicit EOS after EVERY document, even if text ends in EOS
                    key = 'heal' if sums['heal'] < heal_tokens else 'tuning' if sums['tuning'] < tune_rows*length else 'unused'
                    digest = register(text,key)
                    counts[key] += 1; sums[key] += len(tokens); qwen_sum += q
                    ids_by_split[key].append(index)
                    if key != 'unused': writers[key].add(tokens)
                    docs.write(json.dumps(dict(split='train',index=index,group=key,text_sha256=digest,
                        qwen_tokens=q,gpt_neox_tokens=len(tokens)))+'\n')
                    if qwen_sum >= reserved_qwen and reserved_prefix is None:
                        reserved_prefix = sum(counts[k] for k in ('heal','tuning','unused'))
                    enough = sums['heal'] >= heal_tokens and sums['tuning'] >= tune_rows*length
                    if reserved_prefix is not None and enough:
                        extended = sum(counts[k] for k in ('heal','tuning','unused')) > reserved_prefix
                        finished = True; break
                now = time.monotonic()
                if now-last_progress >= 30 or finished:
                    progress = dict(stage='tokenizing_train', counts=counts, document_tokens=sums,
                        qwen_tokens=qwen_sum, elapsed_seconds=now-started)
                    write_json(root/'progress.json', progress)
                    print('A2_DATA_PROGRESS', json.dumps(progress), flush=True)
                    last_progress = now
                if finished: break
            if not finished: raise ValueError('Training split cannot meet reserved/heal/tuning budgets')
            for start in range(0,len(validation),batch_size):
                indices = range(start,min(start+batch_size,len(validation)))
                texts = [validation[i]['text'] for i in indices]
                neo = tokenizer(texts,add_special_tokens=False,truncation=False)['input_ids']
                for index,text,n in zip(indices,texts,neo,strict=True):
                    # Unused reserved documents are not part of the T7 disjointness test.
                    digest = register(text,'final'); tokens = list(n)+[0]
                    writers['final'].add(tokens); counts['final'] += 1; sums['final'] += len(tokens)
                    ids_by_split['final'].append(index)
                    docs.write(json.dumps(dict(split='validation',index=index,group='final',
                        text_sha256=digest,gpt_neox_tokens=len(tokens)))+'\n')
    except BaseException:
        for writer in writers.values():
            writer.tokens.close(); writer.starts.close()
        raise
    streams = {key:writer.close() for key,writer in writers.items()}
    small = sorted(np.random.default_rng(2029).choice(tune_rows,small_rows,replace=False).tolist())
    np.savez(root/'indices.npz', **{k:np.asarray(v,dtype=np.int64) for k,v in ids_by_split.items()}, small=small)
    result = dict(status='complete', provenance=provenance, counts=counts, document_tokens=sums,
        source_counts={'train':len(train),'validation':len(validation)},
        streams=streams, small_indices=small, reserved_qwen_tokens=qwen_sum,
        qwen_count_convention='add_special_tokens=False plus one document EOS',
        qwen_counts_source='verified_published_index' if indexed_counts else 'tokenizer',
        gpt_neox_qwen_ratio=sum(sums[k] for k in ('heal','tuning','unused'))/qwen_sum,
        original_reserved_prefix=reserved_prefix, extended=extended,
        documents_sha256=sha256(root/'documents.jsonl'), indices_sha256=sha256(root/'indices.npz'),
        permutation_seed=2028, small_seed=2029, disjoint_by_id_and_content=True)
    write_json(root/'manifest.json',result)
    print('A2_DATA_COMPLETE', json.dumps(dict(counts=counts, streams=streams,
        elapsed_seconds=time.monotonic()-started)), flush=True)
    return result


def prepare_secondary(dataset,tokenizer,output,*,provenance,length=4096,rows=256,batch_size=64):
    """Fixed public shard, stored order, first 256 packed sequences; never selects arms."""
    if tokenizer.eos_token_id!=0:raise ValueError('GPT-NeoX EOS must be zero')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    writer=PackedWriter(root,'final',length,rows)
    try:
        for start in range(0,len(dataset),batch_size):
            texts=[dataset[i]['text'] for i in range(start,min(start+batch_size,len(dataset)))]
            for ids in tokenizer(texts,add_special_tokens=False,truncation=False)['input_ids']:
                writer.add(list(ids)+[0])
            if writer.count>=rows*length:break
        if writer.count<rows*length:raise ValueError('Public shard has too few tokens')
        stream=writer.close()
    finally:
        writer.tokens.close();writer.starts.close()
    result=dict(status='complete',streams={'final':stream},provenance=provenance,
        overlap_caveat='May overlap pretraining, more for longer-trained parents; no decision attached')
    write_json(root/'manifest.json',result)
    return result


def training_order(count,suffix=''):
    """A retry changes LR only, including the T14b order-noise experiment."""
    return np.random.default_rng(1).permutation(count) if suffix.startswith('-t14b') else np.arange(count)
