"""Frozen training-frequency counts, packed evaluation blocks and probe IDs."""
import argparse
import hashlib
import json
from pathlib import Path

from eval.runtime import offline
offline()
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from capacity_allocation.data import load_text_data, preprocess_text, sha256, write_json
from eval.runtime import languages
from scripts.verify_manifest import create, verify

BUCKETS = ('head', 'common', 'mid', 'rare', 'tail', 'unseen', 'eos', 'padding')


def tokenizer_identity(tokenizer):
    # Padding configuration may be changed by the harness; it is not vocabulary
    # or tokenization identity. Preserve normalization/pretokenization/postprocess.
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend.pop('padding', None)
    backend.pop('truncation', None)
    payload = dict(backend=backend, eos=tokenizer.eos_token_id, pad=tokenizer.pad_token_id)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def frequency_buckets(counts, eos_id=None, pad_id=None):
    counts = np.asarray(counts)
    if counts.ndim != 1 or counts.dtype.kind not in 'iu' or np.any(counts < 0):
        raise ValueError('Nonnegative integer counts required')
    mapping = np.full(len(counts), BUCKETS.index('unseen'), dtype=np.int16)
    excluded = {x for x in (eos_id, pad_id) if x is not None}
    if any(x < 0 or x >= len(counts) for x in excluded):
        raise ValueError('Special token outside vocabulary')
    seen = np.array([i for i in np.flatnonzero(counts) if i not in excluded], dtype=np.int64)
    seen = seen[np.lexsort((seen, -counts[seen].astype(np.int64)))]
    # Ceil boundaries: deterministic, nonoverlapping; tiny vocabularies can have
    # empty buckets. Never force nonempty groups by moving membership per model.
    edges = [0] + [(len(seen)*p+99)//100 for p in (1, 10, 50, 90, 100)]
    for bucket, (start, end) in enumerate(zip(edges, edges[1:])):
        mapping[seen[start:end]] = bucket
    if pad_id is not None:
        mapping[pad_id] = BUCKETS.index('padding')
    if eos_id is not None:  # Shared EOS/pad ID: actual EOS is not padding.
        mapping[eos_id] = BUCKETS.index('eos')
    return mapping


def count_tokens(blocks, vocab_size, batch_size=512):
    if vocab_size < 1 or batch_size < 1:
        raise ValueError('Positive vocabulary and counting batch required')
    counts = np.zeros(vocab_size, dtype=np.int64)
    for batch in blocks.with_format('numpy').iter(batch_size=batch_size):
        ids = np.asarray(batch['input_ids']).reshape(-1)
        if ids.dtype.kind not in 'iu' or ids.min() < 0 or ids.max() >= vocab_size:
            raise ValueError('Training IDs outside model vocabulary')
        counts += np.bincount(ids, minlength=vocab_size)
    return counts


def prepare(train_dir, eval_dir, tokenizer_path, output_dir, *, vocab_size,
            selected_languages='en', block_size=2048, workers=160,
            map_batch_size=1000, cache_dir=None, probe_blocks=8, seed=42, training_shuffle_seed=42):
    if Path(train_dir).resolve() == Path(eval_dir).resolve() or probe_blocks < 1:
        raise ValueError('Separate train/eval paths and positive probe size required')
    selected = languages(selected_languages)
    tokenizer_path = Path(tokenizer_path).resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    if not tokenizer.is_fast or tokenizer.eos_token_id is None or len(tokenizer) > vocab_size:
        raise ValueError('Compatible fast training tokenizer with EOS required')
    tokenizer.model_max_length = 10**30
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    options = dict(block_size=block_size, num_proc=workers, batch_size=map_batch_size, cache_dir=cache_dir)
    raw = load_text_data(train_dir, selected)
    train = preprocess_text(raw, tokenizer, **options)
    counts = count_tokens(train, vocab_size)
    mapping = frequency_buckets(counts, tokenizer.eos_token_id, tokenizer.pad_token_id)
    with (root/'frequencies.npz').open('xb') as handle:
        np.savez(handle, counts=counts, token_to_bucket=mapping)
    rng = np.random.default_rng(seed)
    validation, probes = {}, {}
    for lang in selected:
        source = load_text_data(eval_dir, [lang])
        blocks = preprocess_text(source, tokenizer, **options)
        # A small immutable copy, not a second copy of the training corpus.
        blocks.with_format(None).save_to_disk(str(root/'eval'/lang))
        validation[lang] = dict(raw_fingerprint=source._fingerprint,
            packed_fingerprint=blocks._fingerprint, blocks=len(blocks),
            scored_targets=len(blocks)*(block_size-1))
        if len(blocks) < probe_blocks:
            raise ValueError(f'Not enough {lang} blocks for requested fixed probe')
        probes[lang] = sorted(rng.choice(len(blocks), probe_blocks, replace=False).tolist())
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file())
    manifest = create(root, files)
    manifest.update(success=True, schema='capacity_diagnostics_v1', languages=selected,
        vocab_size=vocab_size, tokenizer_sha256=tokenizer_identity(tokenizer),
        tokenizer_files={p.name: sha256(p) for p in tokenizer_path.iterdir()
                         if p.is_file() and p.name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')},
        eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id,
        bucket_names=list(BUCKETS), bucket_rule='seen_non_special_types_count_desc_id_asc_ceil_percentiles',
        frequency_scope='full_selected_packed_training_pool_not_checkpoint_consumed_prefix',
        training=dict(raw_fingerprint=raw._fingerprint, documents=len(raw),
            packed_fingerprint=train._fingerprint, blocks=len(train),
            shuffled_fingerprint=train.shuffle(seed=training_shuffle_seed)._fingerprint,
            shuffle_seed=training_shuffle_seed,
            processed_tokens=int(counts.sum()), scored_targets=len(train)*(block_size-1)),
        validation=validation, probe_ids=probes, probe_seed=seed,
        preprocessing=dict(block_size=block_size, workers=workers, map_batch_size=map_batch_size,
                           add_special_tokens=False, insert_eos=False, drop_batch_tails=True),
        code={str(p.relative_to(Path(__file__).resolve().parents[1])): sha256(p)
              for p in (Path(__file__).resolve(), Path(__file__).resolve().parents[1]/'capacity_allocation/data.py')})
    write_json(root/'manifest.json', manifest)
    return manifest


def load_bundle(root, tokenizer=None, vocab_size=None):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('success') is not True or manifest.get('schema') != 'capacity_diagnostics_v1':
        raise ValueError('Incomplete/unsupported diagnostic manifest')
    verify(root, manifest)
    if manifest.get('bucket_names') != list(BUCKETS):
        raise ValueError('Unknown bucket convention')
    if (tokenizer is not None and tokenizer_identity(tokenizer) != manifest['tokenizer_sha256']) or (
            vocab_size is not None and vocab_size != manifest['vocab_size']):
        raise ValueError('Diagnostic tokenizer/vocabulary mismatch')
    with np.load(root/'frequencies.npz', allow_pickle=False) as data:
        counts, mapping = data['counts'], data['token_to_bucket']
    expected = frequency_buckets(counts, manifest['eos_id'], manifest['pad_id'])
    if len(counts) != manifest['vocab_size'] or not np.array_equal(expected, mapping):
        raise ValueError('Frequency/bucket mapping mismatch')
    if int(counts.sum()) != manifest['training']['processed_tokens']:
        raise ValueError('Training count coverage mismatch')
    datasets = {}
    for lang in languages(manifest['languages']):
        data = load_from_disk(str(root/'eval'/lang)).with_format('torch')
        if len(data) != manifest['validation'][lang]['blocks']:
            raise ValueError('Diagnostic eval coverage mismatch')
        ids = manifest['probe_ids'][lang]
        if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < len(data) for i in ids):
            raise ValueError('Invalid frozen probe indices')
        datasets[lang] = data
    return manifest, counts, mapping, datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train-dir', 'eval-dir', 'tokenizer-name', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--vocab-size', type=int, default=151936)
    parser.add_argument('--languages', default='en')
    parser.add_argument('--block-size', type=int, default=2048)
    parser.add_argument('--preprocessing-num-workers', type=int, default=160)
    parser.add_argument('--preprocessing-batch-size', type=int, default=1000)
    parser.add_argument('--preprocessing-cache-dir')
    parser.add_argument('--probe-blocks', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--training-shuffle-seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.train_dir, args.eval_dir, args.tokenizer_name, args.output_dir,
        vocab_size=args.vocab_size, selected_languages=args.languages, block_size=args.block_size,
        workers=args.preprocessing_num_workers, map_batch_size=args.preprocessing_batch_size,
        cache_dir=args.preprocessing_cache_dir, probe_blocks=args.probe_blocks, seed=args.seed,
        training_shuffle_seed=args.training_shuffle_seed)
    print('DIAGNOSTIC_BUNDLE_COMPLETE', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
