"""Frozen document split and exact concatenated stream, using ordinary HF caches.

No dataset download or exported token corpus. The manifest contains document IDs
and tokenizer/revision identity; all arms reuse the same local text and caches.
"""
import json
from pathlib import Path

from datasets import Dataset
import pyarrow.compute as pc

from training_data import digest, document_end, tokenizer_identity, tokenize_batch, group_texts


def document_split(raw, args, tokenizer):
    if not args.dataset_revision or not args.validation_manifest:
        raise ValueError('document_stream requires dataset_revision and validation_manifest')
    if 'id' not in raw.column_names:
        raise ValueError('document_stream requires original dataset document IDs in column id')
    if raw.data.column('id').null_count or pc.count_distinct(raw.data.column('id')).as_py() != len(raw):
        raise ValueError('Source document IDs must be nonnull and unique')
    if args.document_split not in ('train', 'validation'):
        raise ValueError('document_split must be train or validation')
    if args.validation_tokens < 1:
        raise ValueError('validation_tokens must be positive')
    end = document_end(tokenizer, args.document_end_token)
    identity = dict(revision=args.dataset_revision, tokenizer=tokenizer_identity(tokenizer),
                    document_end={k: end[k] for k in ('token', 'id')},
                    seed=0, minimum_tokens=args.validation_tokens)
    raw = raw.shuffle(seed=0, load_from_cache_file=not args.overwrite_cache)
    path = Path(args.validation_manifest)
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest['identity'] != identity:
            raise ValueError('Validation manifest revision/tokenizer/budget differs; do not overwrite it')
        selected = set(manifest['document_ids'])
        if len(selected) != len(manifest['document_ids']):
            raise ValueError('Validation manifest contains duplicate IDs')
        # A different subset may lack some held-out documents, but must still
        # exclude all IDs it contains. Record split sizes in the data receipt.
    else:
        if args.document_split != 'train':
            raise ValueError('Create the validation manifest from the training source first')
        selected, tokens, ordered = set(), 0, []
        for end_row in range(len(raw), 0, -128):
            rows = raw[max(0, end_row-128):end_row]
            lengths = [len(ids) for ids in tokenize_batch(rows, tokenizer, end['id'])['input_ids']]
            for doc_id, length in zip(reversed(rows['id']), reversed(lengths)):
                selected.add(doc_id)
                ordered.append(doc_id)
                tokens += length
                if tokens >= args.validation_tokens:
                    break
            if tokens >= args.validation_tokens:
                break
        if tokens < args.validation_tokens or len(selected) == len(raw):
            raise ValueError('Not enough text for the requested held-out tokens and a nonempty training split')
        manifest = dict(schema=1, identity=identity, document_ids=ordered,
                        token_count=tokens, num_documents=len(ordered))
        path.parent.mkdir(parents=True, exist_ok=True)
        # main_process_first serializes workers. Never replace an existing split.
        with path.open('x') as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
    validation = args.document_split == 'validation'
    selected_rows = raw.filter(lambda batch: [(doc_id in selected) == validation for doc_id in batch['id']],
                               batched=True, load_from_cache_file=not args.overwrite_cache,
                               new_fingerprint=digest(dict(raw=raw._fingerprint, manifest=manifest,
                                                           split=args.document_split))[:32])
    if not len(selected_rows):
        raise ValueError('Requested document split is empty')
    if validation and len(selected_rows) != manifest['num_documents']:
        raise ValueError('Validation source does not contain the complete frozen document manifest')
    return selected_rows.select_columns(['text']), dict(
        sha256=digest(manifest), split=args.document_split, identity=identity,
        path=str(path.resolve()),
        validation_documents=manifest['num_documents'], validation_tokens=manifest['token_count'])


def packed_stream(tokenized, block_size, batch_size, isolate_documents=True):
    """Carry the remainder across tokenization batches; drop only the final tail."""
    pending = {name: [] for name in ('input_ids', 'attention_mask', 'segments')}
    for batch in tokenized.iter(batch_size=batch_size):
        joined = {key: [pending[key], *batch[key]] for key in pending}
        result = group_texts(joined, block_size, isolate_documents)
        total = {key: [x for row in rows for x in row] for key, rows in joined.items()}
        used = len(result['input_ids'])*block_size
        pending = {key: row[used:] for key, row in total.items()}
        for i in range(len(result['input_ids'])):
            yield {key: values[i] for key, values in result.items()}


def pack_document_stream(tokenized, data_args, block_size):
    return Dataset.from_generator(packed_stream, gen_kwargs=dict(
        tokenized=tokenized, block_size=block_size, batch_size=data_args.map_batch_size,
        isolate_documents=data_args.isolate_documents), cache_dir=data_args.data_cache_dir)
