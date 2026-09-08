"""Saved text -> batched Dataset.map -> cached LM blocks, as in sparse embedding.

No sampling, downloads, SQLite index, or custom binary token format. Packing
matches the old trainer: concatenate within each map batch, drop its short
tail, do not insert EOS, and copy input_ids to labels without masking EOS.
"""
import hashlib
from itertools import chain
import json
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def load_text_data(directory, languages=('en',)):
    """Read old prepare_data.py output: train/en/shard_0000 or eval/en."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not languages or len(set(languages)) != len(languages):
        raise ValueError('Select unique languages explicitly')
    parts = []
    for lang in sorted(languages):
        if Path(lang).name != lang or lang in ('.', '..'):
            raise ValueError('Language must be a directory name')
        language_dir = root / lang
        if not language_dir.is_dir():
            raise FileNotFoundError(language_dir)
        shards = sorted(p for p in language_dir.iterdir()
                        if p.is_dir() and p.name.startswith('shard_'))
        for path in shards or [language_dir]:
            dataset = load_from_disk(str(path))
            if 'text' not in dataset.column_names or not len(dataset):
                raise ValueError(f'Expected a nonempty saved text Dataset: {path}')
            parts.append(dataset)
    return concatenate_datasets(parts)


def group_texts(examples, block_size):
    concatenated = list(chain.from_iterable(examples['input_ids']))
    usable = len(concatenated) // block_size * block_size
    blocks = [concatenated[i:i+block_size] for i in range(0, usable, block_size)]
    return {'input_ids': blocks, 'labels': [ids.copy() for ids in blocks]}


def preprocess_text(raw, tokenizer, *, block_size=2048, num_proc=16,
                    batch_size=1000, cache_dir=None, overwrite_cache=False):
    if block_size < 2 or num_proc < 1 or batch_size < 1:
        raise ValueError('Positive preprocessing settings and block_size >= 2 required')
    workers = min(num_proc, len(raw))
    if not workers:
        raise ValueError('Empty text dataset')
    # Map drops tails per batch/worker, so both settings belong in cache identity.
    key = hashlib.sha256(json.dumps(dict(
        raw=raw._fingerprint, tokenizer=tokenizer.backend_tokenizer.to_str(),
        block_size=block_size, workers=workers, batch_size=batch_size,
        code=sha256(__file__), add_special_tokens=False, insert_eos=False),
        sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    def tokenize_function(examples):
        return {'input_ids': tokenizer(examples['text'], add_special_tokens=False,
                                       truncation=False, return_attention_mask=False)['input_ids']}

    tokenized = raw.map(
        tokenize_function, batched=True, batch_size=batch_size, num_proc=workers,
        remove_columns=raw.column_names, load_from_cache_file=not overwrite_cache,
        cache_file_name=str(cache / f'{key}-tokenized.arrow') if cache else None,
        new_fingerprint=hashlib.sha256((key+'tokenized').encode()).hexdigest()[:32],
        desc='Running tokenizer on dataset')
    blocks = tokenized.map(
        group_texts, fn_kwargs={'block_size': block_size}, batched=True,
        batch_size=batch_size, num_proc=workers,
        load_from_cache_file=not overwrite_cache,
        cache_file_name=str(cache / f'{key}-packed.arrow') if cache else None,
        new_fingerprint=hashlib.sha256((key+'packed').encode()).hexdigest()[:32],
        desc=f'Grouping texts in chunks of {block_size}')
    if not len(blocks):
        raise ValueError('No complete blocks; increase data/map batch size')
    return blocks.with_format('torch')
