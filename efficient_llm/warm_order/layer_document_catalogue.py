"""Read-only original-document mapping for the unshuffled A2 packed streams.

No model loading, network, CUDA, training or output publication. Pins must come
from an independently verified receipt. This does not certify a complete history
of exclusions, near-duplicate removal or current node ownership. Other stream
transforms require their own mapping; never guess offsets from repacked starts.
"""
from bisect import bisect_right
from dataclasses import dataclass
import hashlib
from itertools import islice
import json
from pathlib import Path
import struct

import numpy as np


GROUPS = ('heal', 'tuning', 'final')
RECORD = struct.Struct('<Q32sQB')


def pinned(path, digest):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    if h.hexdigest() != digest:
        raise ValueError('File hash mismatch: '+str(path))


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid integer field')
    return value


@dataclass(frozen=True)
class Span:
    start: int
    stop: int
    text_sha256: str
    source_index: int
    source_row: int
    original_tokens: int

    @property
    def truncated(self):
        return self.stop-self.start != self.original_tokens


def shard_records(receipt_path, index_path, receipt_sha256, source_index):
    pinned(receipt_path, receipt_sha256)
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt['status'] != 'complete' or receipt['source']['index'] != source_index:
        raise ValueError('Shard identity mismatch')
    info = receipt['files']['documents.bin']
    if Path(index_path).stat().st_size != info['bytes'] or info['bytes'] % RECORD.size:
        raise ValueError('Invalid document index size')
    pinned(index_path, info['sha256'])
    source_rows = integer(receipt['rows'])
    totals = {g: dict(documents=0, tokens=0) for g in GROUPS}
    previous = -1
    with Path(index_path).open('rb') as f:
        while raw := f.read(RECORD.size):
            if len(raw) != RECORD.size:
                raise ValueError('Truncated index record')
            row, digest, tokens, code = RECORD.unpack(raw)
            bucket = int.from_bytes(digest[:8], 'big') % 10000
            expected = 'tuning' if bucket < 2 else 'final' if bucket < 10 else 'heal'
            if row <= previous or row >= source_rows or tokens == 0 or code >= len(GROUPS):
                raise ValueError('Invalid document record')
            group = GROUPS[code]
            if group != expected:
                raise ValueError('Text-hash partition mismatch')
            previous = row
            totals[group]['documents'] += 1
            totals[group]['tokens'] += tokens
            yield row, digest.hex(), tokens, group
    if totals != receipt['streams']:
        raise ValueError('Document index totals disagree with receipt')
    if sum(t['documents'] for t in totals.values()) + integer(receipt['empty_documents_skipped']) != source_rows:
        raise ValueError('Source row coverage mismatch')


def packed_spans(manifest_path, manifest_sha256, group, shard_files):
    """shard_files maps index -> (receipt_path, documents.bin path), all local.

    Consumes complete shard indices even when a packed prefix ends mid-document,
    so their receipt totals are still checked. Returns spans in packed order.
    """
    if group not in GROUPS:
        raise ValueError('Unknown group')
    pinned(manifest_path, manifest_sha256)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['status'] != 'complete' or manifest['schema'] != 'a2-new-corpus-v1':
        raise ValueError('Unsupported packed manifest')
    expected = integer(manifest['streams'][group]['tokens'], 1)
    spans, seen, offset = [], set(), 0
    prefixes = [p for p in manifest['source_prefixes'] if p['group'] == group]
    for part in prefixes:
        index, take = integer(part['source_index']), integer(part['tokens'], 1)
        if index in seen:
            raise ValueError('Repeated shard prefix')
        seen.add(index)
        remaining = take
        receipt, documents = shard_files[index]
        for row, digest, count, record_group in shard_records(
                receipt, documents, part['shard_receipt_sha256'], index):
            if record_group == group and remaining:
                used = min(remaining, count)
                spans.append(Span(offset, offset+used, digest, index, row, count))
                offset += used
                remaining -= used
        if remaining:
            raise ValueError('Packed prefix exceeds shard coverage')
    if offset != expected:
        raise ValueError('Packed token total mismatch')
    return tuple(spans)


def load_row_order(path, digest, expected_rows):
    """E2 destination row r reads source row order[r], not the inverse."""
    integer(expected_rows, 1)
    pinned(path, digest)
    order = np.load(path, allow_pickle=False)
    if not isinstance(order, np.ndarray) or order.ndim != 1 or order.dtype.kind not in 'iu':
        raise ValueError('Invalid row-order array')
    if len(order) != expected_rows or not np.array_equal(np.sort(order), np.arange(expected_rows)):
        raise ValueError('Order is not a full prefix permutation')
    order.flags.writeable = False
    return order


def verify_original_starts(spans, path, digest):
    """Check ORIGINAL boundaries with chunk-bounded comparison storage.

    The caller still holds the full catalogue. This avoids constructing another
    whole-corpus Python list of observed and expected boundary positions.
    """
    pinned(path, digest)
    if not spans or Path(path).stat().st_size != spans[-1].stop:
        raise ValueError('Original starts length differs')
    if spans[0].start != 0 or any(
            s.stop <= s.start or (i and spans[i-1].stop != s.start)
            for i, s in enumerate(spans)):
        raise ValueError('Noncontiguous catalogue')
    data = np.memmap(path, dtype='u1', mode='r')
    expected = (s.start for s in spans)
    for offset in range(0, len(data), 1 << 20):
        block = data[offset:offset+(1 << 20)]
        if np.any(block > 1):
            raise ValueError('Nonbinary boundary marker')
        observed = np.flatnonzero(block) + offset
        wanted = np.fromiter(islice(expected, len(observed)), dtype=np.int64)
        if not np.array_equal(observed, wanted):
            raise ValueError('Original boundary positions disagree with document catalogue')
    if next(expected, None) is not None:
        raise ValueError('Original boundary positions disagree with document catalogue')


def iter_mapped_rows(spans, row_ids, length=512, order=None):
    """Yield every source document touched, including cross-document windows.

    Repeated historical rows are allowed. This API covers direct packed rows or
    a checked E2 prefix permutation only; no arbitrary inferred offset mapping.
    """
    integer(length, 1)
    if not spans or spans[0].start != 0 or any(
            s.stop <= s.start or (i and spans[i-1].stop != s.start)
            for i, s in enumerate(spans)):
        raise ValueError('Noncontiguous catalogue')
    starts = [s.start for s in spans]
    for row in row_ids:
        # Historical .npy/.npz selections yield numpy integer scalars. Keep
        # metadata integer() strict, and reject floats/bools rather than cast
        # them into a different row silently.
        if isinstance(row, np.integer) and not isinstance(row, np.bool_):
            row = int(row)
        integer(row)
        if order is not None and row >= len(order):
            raise ValueError('Destination row out of range')
        source_row = int(order[row]) if order is not None else row
        lo, hi = source_row*length, (source_row+1)*length
        if lo < 0 or hi > spans[-1].stop:
            raise ValueError('Source window outside packed coverage')
        first = bisect_right(starts, lo)-1
        touched = []
        for i in range(first, len(spans)):
            if spans[i].start >= hi:
                break
            touched.append(spans[i])
        yield dict(row_id=row, source_row=source_row,
            document_keys=sorted(set(s.text_sha256 for s in touched)),
            single_document_occurrence=len(touched) == 1,
            touches_truncated_document=any(s.truncated for s in touched))


def map_rows(spans, row_ids, length=512, order=None):
    """Materialize small selections; use iter_mapped_rows for large histories.

    Both APIs require a checked catalogue and, if supplied, a checked order.
    The iterator can raise after yielding a prefix; consumers must not publish
    a completed exclusion proof until the entire input has been consumed.
    """
    return list(iter_mapped_rows(spans, row_ids, length=length, order=order))
