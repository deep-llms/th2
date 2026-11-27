"""Rebuild the DOT E2 training stream on a local machine (B200 restart, 26 Nov 2026).

E2 (docs/A2_DOT_DATA_REBUILD_20261118.md) is the first 2,000,000,000 tokens of the packed DCLM heal stream
(resources/a2_corpora_30b_20261029.json: pinned shard order, GPT-NeoX tokenizer, no BOS, EOS 0, SHA256 text
partition), cut into 512-token rows and permuted with numpy default_rng(0). The heal stream is the shards' heal
tokens concatenated in file order, so only the first shards are needed. This script reuses the original code
(scripts/a2_prepare_corpora.shard_worker and scripts/a2_dot_rebuild_data.repack_e2) and checks every output hash
against resources/a2_dot_data_ohmv_20261118.json dot_e2_expected['train'].

Usage: python scripts/local_dot_e2_rebuild.py --root /disk/thuat/dot_data_20261126 --shards 20 --workers 8
(raw/dclm/<index>.parquet must be downloaded and are re-verified; tokenizer files in <root>/tokenizer).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.a2_dot_rebuild_data import repack_e2  # noqa: E402
from scripts.a2_prepare_corpora import sha, shard_worker, verify_file, write_json  # noqa: E402

E2_TOKENS = 2_000_000_000


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--shards', type=int, default=20)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    recipe = json.loads((ROOT/'resources/a2_corpora_30b_20261029.json').read_text())
    expected = json.loads((ROOT/'resources/a2_dot_data_ohmv_20261118.json').read_text())['dot_e2_expected']['train']
    tok_dir = a.root/'tokenizer'
    for name, want in recipe['tokenizer_files'].items():
        if sha(tok_dir/name) != want:
            raise ValueError('Tokenizer hash differs: ' + name)
    tok_sha = recipe['tokenizer_files']['tokenizer.json']
    files = recipe['corpora']['dclm']['files'][:a.shards]
    deadline = time.monotonic() + 6 * 3600
    tasks = []
    for record in files:
        record = dict(record, local=str(a.root/'raw'/'dclm'/f"{record['index']:05d}.parquet"))
        tasks.append((record, str(tok_dir/'tokenizer.json'), tok_sha, str(a.root/'shards'/f"{record['index']:05d}"), deadline))
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(shard_worker, tasks))
    heal = [r['streams']['heal']['tokens'] for r in results]
    print('HEAL_TOKENS_PER_SHARD', heal, 'cumulative', sum(heal), flush=True)
    if sum(heal) < E2_TOKENS:
        raise ValueError(f'Need more shards: {sum(heal)} heal tokens < {E2_TOKENS}')
    packed = a.root/'packed'/'dclm'
    packed.mkdir(parents=True, exist_ok=False)
    count, used = 0, []
    with (packed/'heal.bin').open('xb') as ids, (packed/'heal.starts').open('xb') as starts:
        for (record, *_), saved in zip(tasks, results):
            take = min(E2_TOKENS - count, saved['streams']['heal']['tokens'])
            if not take:
                break
            root = Path(_[2])
            for suffix in ('.bin', '.starts'):
                verify_file(root/('heal' + suffix), saved['files']['heal' + suffix])
            for suffix, width, out in (('.bin', 4, ids), ('.starts', 1, starts)):
                remaining = take * width
                with (root/('heal' + suffix)).open('rb') as f:
                    while remaining:
                        b = f.read(min(8 << 20, remaining))
                        if not b:
                            raise ValueError('Short shard stream')
                        out.write(b)
                        remaining -= len(b)
            used.append(dict(source_index=record['index'], tokens=take))
            count += take
    if count != E2_TOKENS:
        raise ValueError('Heal prefix incomplete')
    proof = repack_e2(packed, a.root/'dot'/'e2', limits=[('train', 'heal', E2_TOKENS)])['train']
    checks = {k: (proof[k], expected[k]) for k in ('rows', 'tokens', 'source_document_starts', 'maximum_id',
                                                   'order_sha256', 'token_sha256', 'starts_sha256')}
    ok = all(x == y for x, y in checks.values())
    write_json(a.root/'dot'/'e2_local_rebuild.json', dict(ok=ok, checks=checks, heal_prefix_sources=used,
                                                          heal_tokens_per_shard=heal, recipe='resources/a2_corpora_30b_20261029.json'))
    print('E2_REBUILD', 'MATCH' if ok else 'MISMATCH', json.dumps(checks), flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
