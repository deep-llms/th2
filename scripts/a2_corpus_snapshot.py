"""Pack a fixed training prefix from immutable completed corpus shards (CPU only)."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.a2_prepare_corpora import copy_prefix, sha, tokenizer_identity, verify_file, verify_packed, write_json


def prefix_digest(parts, suffix, bytes_per_token):
    digest = hashlib.sha256()
    for root, take, _, _ in parts:
        remaining = take * bytes_per_token
        with (root / ('heal' + suffix)).open('rb') as source:
            while remaining:
                block = source.read(min(8 << 20, remaining))
                if not block:
                    raise ValueError('Short source prefix')
                digest.update(block)
                remaining -= len(block)
    return digest.hexdigest()


def snapshot(plan_path, corpus, output, tokens):
    plan_path, output = Path(plan_path), Path(output)
    plan = json.loads(plan_path.read_text())
    length = plan['length']
    target = tokens // length * length
    if target <= 0 or output.exists():
        raise ValueError('Positive budget and fresh snapshot output required')
    token_sha = tokenizer_identity(plan)
    parts, available = [], 0
    for record in plan['corpora'][corpus]['files']:
        root = Path(plan['root']) / 'parts' / corpus / f"{record['index']:05d}"
        receipt = root / 'complete.json'
        if not receipt.is_file():
            raise ValueError('Contiguous completed source prefix not yet available')
        raw = receipt.read_bytes()
        saved = json.loads(raw)
        if (saved['status'] != 'complete' or saved['source'] != record
                or saved['tokenizer_sha256'] != token_sha):
            raise ValueError('Completed shard identity differs')
        take = min(target - available, saved['streams']['heal']['tokens'])
        for suffix in ('.bin', '.starts'):
            verify_file(root / ('heal' + suffix), saved['files']['heal' + suffix])
        parts.append((root, take, sha(receipt), record['index']))
        available += take
        if available == target:
            break
    if available != target:
        raise ValueError('Insufficient completed training tokens')
    expected_tokens = prefix_digest(parts, '.bin', 4)
    expected_starts = prefix_digest(parts, '.starts', 1)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'heal.bin').open('xb') as ids, (output / 'heal.starts').open('xb') as starts:
        for root, take, _, _ in parts:
            copy_prefix(root / 'heal.bin', ids, take * 4)
            copy_prefix(root / 'heal.starts', starts, take)
    manifest = dict(status='complete', schema='a2-training-prefix-v1', corpus=corpus,
        plan_sha256=sha(plan_path), tokenizer_sha256=token_sha, eos_id=plan['eos_id'],
        streams={'heal': dict(rows=target // length, tokens=target, length=length,
            token_sha256=expected_tokens, starts_sha256=expected_starts)},
        source_prefixes=[dict(index=i, tokens=n, shard_receipt_sha256=h) for _, n, h, i in parts],
        evaluation='Training stream only. Select a separately frozen validation set in the experiment protocol.',
        order='Sequential rows, no shuffle; exact prefix of the full corpus training stream.')
    verify_packed(output, manifest)
    # Reader smoke uses the same schema; publication remains atomic.
    write_json(output / 'manifest.json', manifest)
    from efficient_llm.a2.data import PackedRows
    rows = PackedRows(output, 'heal', count=min(8, target // length), verify=False)
    assert len(rows[0]['input_ids']) == length
    assert len(rows.ids[-1]) == length
    write_json(output / 'complete.json', dict(status='complete', manifest_sha256=sha(output / 'manifest.json'), native_reader_passed=True, source_prefix_hashes_match=True))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--corpus', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--tokens', required=True, type=int)
    a = p.parse_args()
    result = snapshot(a.plan, a.corpus, a.output, a.tokens)
    print('CORPUS_SNAPSHOT_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
