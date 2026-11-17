"""CPU-only OHMV reconstruction from historical A2/DOT recipes.

Downloads are controller #d operations only. This worker reads local files,
retains partial outputs on failure, and publishes READY only after old hashes
match. Provenance manifests get new paths/timings; historical files stay intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
from importlib.metadata import version

from scripts.a2_prepare_corpora import sha, write_json, verify_file

RECIPE = 'resources/a2_dot_data_ohmv_20261118.json'


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def recipe():
    value = read(RECIPE)
    for name, digest in value['locks'].items():
        require(sha(name) == digest, 'Historical source changed: ' + name)
    return value


def plan_for(cfg, corpus):
    plan = read(cfg['plan'])
    root = Path(cfg['data_root'])
    plan['root'] = str(root/'corpora'/corpus)
    plan['tokenizer_path'] = str(root/'tokenizer')
    selected = plan['corpora'][corpus]
    selected['files'] = selected['files'][:cfg['raw_counts'][corpus]]
    for item in selected['files']:
        item['local'] = str(root/'raw'/corpus/f"{item['index']:05d}.parquet")
    plan['corpora'] = {corpus: selected}
    return plan


def raw_records(cfg):
    return [(corpus, item) for corpus in cfg['raw_counts']
            for item in plan_for(cfg, corpus)['corpora'][corpus]['files']]


def download_command(cfg, batch):
    """Dev-side rendering only: <=10 raw shards, no credentials or weights."""
    entries = raw_records(cfg)
    require(0 <= batch < (len(entries)+9)//10, 'Invalid batch number')
    lines = [f'#d +a # ohmv-a2-dot-data-batch{batch+1:02d}-20261118']
    if batch == 0:
        old = read('resources/a2_inputs_20261028.json')
        provenance = old['tokenizer_provenance']['gpt_neox']
        lines.append('#models')
        for name in provenance['files']:
            url = f"https://huggingface.co/{provenance['repo']}/resolve/{provenance['revision']}/{name}"
            lines.append(f"--url {url} {cfg['data_root']}/tokenizer/{name}")
    lines.append('#datasets')
    for _, item in entries[batch*10:(batch+1)*10]:
        lines.append('--url '+item['url']+' '+item['local'])
    return '\n'.join(lines)+'\n'


def compare(actual, expected, keys):
    for key in keys:
        require(actual[key] == expected[key], 'Historical content differs: '+key)


def verify_corpus(actual, expected):
    compare(actual, expected, ['streams', 'tokenizer_sha256', 'eos_id',
                              'small_indices', 'partition', 'packing'])
    prefix = lambda m: [(x['group'], x['source_index'], x['tokens'])
                        for x in m['source_prefixes']]
    require(prefix(actual) == prefix(expected), 'Historical source prefix differs')


def verify_generated(root, expected):
    for name, item in expected.items():
        verify_file(Path(root)/name, item)


def prepare_kg(cfg, output):
    # Original generator implementations, not reimplemented sampling algorithms.
    from efficient_llm.dot_pool.data import make_kg
    from efficient_llm.dot_pool.rl import make_data
    root = Path(cfg['data_root'])/'dot'
    make_kg(root/'e1')
    verify_generated(root/'e1', cfg['dot_kg_expected_files'])
    make_data(root/'rl')
    historical = read(cfg['dot_rl_expected'])
    verify_generated(root/'rl', historical['files'])
    compare(read(root/'rl/manifest.json'), historical, ['counts', 'seed', 'split_order'])
    write_json(output/'kg_verified.json', dict(status='verified', e1=str(root/'e1'),
        e1b_reuses_e1=True, rl=str(root/'rl'), e1_files=len(cfg['dot_kg_expected_files']),
        rl_files=len(historical['files']), file_content_hashes_equal=True))


def repack_e2(source, destination, limits=None):
    """Same prefix/row-permutation/boundary rules as historical dot_pool_prepare."""
    import numpy as np
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    limits = limits or [('train', 'heal', 2_000_000_000), ('val', 'final', 10_000_000)]
    proof = {}
    for dest, name, limit in limits:
        count = limit//512
        ids = np.memmap(source/(name+'.bin'), mode='r', dtype='<u4').reshape(-1, 512)
        starts = np.memmap(source/(name+'.starts'), mode='r', dtype='u1').reshape(-1, 512)
        require(ids.shape == starts.shape and len(ids) >= count, 'Invalid E2 source coverage')
        order = np.random.default_rng(0).permutation(count)
        np.save(destination/(dest+'_order.npy'), order)
        boundaries = maximum = 0
        with (destination/(dest+'.bin')).open('xb') as ft, (destination/(dest+'.starts')).open('xb') as fs:
            for first in range(0, count, 4096):
                ix = order[first:first+4096]
                a, b = np.array(ids[ix]), np.array(starts[ix])
                require(not np.any(b[:, 1:].astype(bool) & (a[:, :-1] != 0)), 'Invalid original EOS boundary')
                boundaries += int(b.sum())
                maximum = max(maximum, int(a.max()))
                b[:, 0] = 1
                a.tofile(ft); b.tofile(fs)
        require(maximum < 50277, 'Token outside GPT-NeoX vocabulary')
        proof[dest] = dict(rows=count, tokens=count*512, dropped_to_512=limit-count*512,
            source_document_starts=boundaries, maximum_id=maximum,
            order_sha256=sha(destination/(dest+'_order.npy')),
            token_sha256=sha(destination/(dest+'.bin')),
            starts_sha256=sha(destination/(dest+'.starts')))
    return proof


def full_and_prefix(path, length):
    full, prefix, offset = hashlib.sha256(), hashlib.sha256(), 0
    with Path(path).open('rb') as handle:
        while block := handle.read(8 << 20):
            full.update(block)
            if offset < length:
                prefix.update(block[:length-offset])
            offset += len(block)
    return dict(full_sha256=full.hexdigest(), prefix_sha256=prefix.hexdigest(), bytes=offset)


def work(cfg):
    from scripts.a2_prepare_corpora import prepare, tokenizer_identity
    from scripts.a2_stage1_data import prepare as shuffle
    output = Path(cfg['output'])
    root = Path(cfg['data_root'])
    require(socket.gethostname() == cfg['host'], 'Wrong node')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only worker required')
    require(os.environ.get('DOT_POOL_SPEC_SHA256') is None, 'Unexpected DOT protocol override')
    for package, pin in [('numpy', '2.4.6'), ('tokenizers', '0.22.1'), ('pyarrow', '26.0.0')]:
        require(version(package) == pin, 'Runtime pin differs: '+package)
    subprocess.run([sys.executable, '-m', 'unittest', 'tests.a2.test_data_rebuild', '-v'], check=True)
    prepare_kg(cfg, output)
    base = plan_for(cfg, 'dclm')
    until = time.monotonic()+168*3600
    while not all((root/'tokenizer'/name).is_file() for name in base['tokenizer_files']):
        require(time.monotonic() < until, 'Tokenizer download deadline exceeded')
        write_json(output/'status.json', dict(stage='waiting_for_tokenizer', utc=time.time()))
        time.sleep(30)
    # Do not treat a still-writing file as a complete tokenizer. Hash admission
    # waits for expected bytes, bounded by the same data-download deadline.
    while True:
        try:
            tokenizer_identity(base)
            break
        except ValueError:
            require(time.monotonic() < until, 'Tokenizer hash never matched')
            write_json(output/'status.json', dict(stage='waiting_for_tokenizer_hash', utc=time.time()))
            time.sleep(30)
    results = {}
    for corpus in ('dclm', 'fineweb_edu'):
        plan = plan_for(cfg, corpus)
        plan_path = output/(corpus+'_plan.json')
        write_json(plan_path, plan)
        write_json(output/'status.json', dict(stage='corpus', corpus=corpus, utc=time.time()))
        prepare(plan_path, deadline_hours=168)
        packed = Path(plan['root'])/'packed'/corpus
        actual = read(packed/'manifest.json')
        old = read(cfg['historical_dclm'])
        expected = old['corpus_expected'] if corpus == 'dclm' else read(cfg['historical_fwe'])
        verify_corpus(actual, expected)
        result = dict(packed=str(packed), packed_manifest_sha256=sha(packed/'manifest.json'),
                      historical_packed_content_equal=True)
        write_json(output/(corpus+'_packed_verified.json'), result)
        if corpus == 'dclm':
            write_json(output/'status.json', dict(stage='dot_e2', utc=time.time()))
            dot = root/'dot/e2'
            proof = repack_e2(packed, dot)
            for name, info in proof.items():
                compare(info, cfg['dot_e2_expected'][name], list(info))
            write_json(dot/'manifest.json', dict(status='complete', streams=proof,
                source_manifest_sha256=sha(packed/'manifest.json'),
                tokenizer_sha256=actual['tokenizer_sha256'], eos_id=0,
                partition=actual['partition'], historical_content_hashes_equal=True))
            write_json(output/'dot_e2_verified.json', dict(status='verified', path=str(dot), streams=proof))
        shuffled = root/'shuffle2030'/corpus
        write_json(output/'status.json', dict(stage='shuffle2030', corpus=corpus, utc=time.time()))
        shuffle(plan_path, sha(packed/'manifest.json'), shuffled, 'docs/A2_stage1_spec.md',
                corpus=corpus, replication_spec='docs/A2_stage1b_fwe_spec.md' if corpus == 'fineweb_edu' else None)
        expected_shuffle = old['shuffle_expected'] if corpus == 'dclm' else read(cfg['historical_fwe_shuffle'])
        actual_shuffle = read(shuffled/'manifest.json')
        compare(actual_shuffle, expected_shuffle, ['streams', 'tokenizer_sha256', 'eos_id',
            'document_count', 'documents_sha256', 'document_order_sha256', 'heldout_documents_sha256',
            'excluded_original_tail_tokens', 'dropped_after_shuffle'])
        if corpus == 'dclm':
            for name, key, size in [('heal.bin', 'tokens', 781248*4096*4),
                                    ('heal.starts', 'boundaries', 781248*4096)]:
                require(full_and_prefix(shuffled/name, size) == old['historical_dt6'][key], 'Depth prefix differs')
        result.update(shuffled=str(shuffled), shuffled_manifest_sha256=sha(shuffled/'manifest.json'),
                      historical_shuffled_content_equal=True)
        results[corpus] = result
        write_json(output/(corpus+'_verified.json'), result)
    complete = dict(status='verified', ready=True, corpora=results, kg=read(output/'kg_verified.json'),
        dot_e2=read(output/'dot_e2_verified.json'), data_root=str(root), training_launched=False,
        note='New provenance paths; original file content verified. CulturaX and SmolLM3 retokenization are separate preparations.')
    write_json(output/'complete.json', complete)
    write_json(output/'status.json', dict(stage='complete', utc=time.time()))
    print('A2_DOT_DATA_COMPLETE', json.dumps(complete), flush=True)


def launch(cfg, commit, source_manifest):
    require(socket.gethostname() == cfg['host'], 'Wrong node')
    require(len(commit) == 40 and set(commit) <= set('0123456789abcdef'), 'Source revision required')
    output = Path(cfg['output'])
    require(not output.exists(), 'Refuse duplicate launch/output')
    require(shutil.disk_usage('/mnt/local').free > 2_000_000_000_000, 'Need 2TB free')
    require(subprocess.run(['tmux', 'has-session', '-t', cfg['session']], capture_output=True).returncode != 0,
            'Session already exists')
    files = read(source_manifest)['files']
    for name, digest in files.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe source member')
        require(sha(name) == digest, 'Source changed: '+name)
    frozen = output/'source'
    for name in files:
        dest = frozen/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, dest)
        require(sha(dest) == files[name], 'Frozen source mismatch')
    write_json(output/'source-manifest.json', dict(commit=commit, files=files, cpu_only=True))
    env = ['env', 'CUDA_VISIBLE_DEVICES=', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
           'HF_DATASETS_OFFLINE=1', 'HF_HUB_DISABLE_TELEMETRY=1', 'PYTHONDONTWRITEBYTECODE=1',
           'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
           'TOKENIZERS_PARALLELISM=false', 'RAYON_NUM_THREADS=1', 'PYTHONPATH='+str(frozen)]
    command = 'exec '+shlex.join(env+['nice', '-n', '10', sys.executable, '-u', '-m',
                'scripts.a2_dot_rebuild_data', 'work'])+' >'+shlex.quote(str(output/'prepare.log'))+' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', cfg['session'], '-c', str(frozen), command], check=True)
    print('A2_DOT_DATA_LAUNCHED', json.dumps(dict(output=str(output), session=cfg['session'], commit=commit)), flush=True)


def health(cfg):
    output = Path(cfg['output'])
    result = {'host':socket.gethostname(), 'free_bytes':shutil.disk_usage('/mnt/local').free, 'raw':{}}
    for corpus in cfg['raw_counts']:
        entries = plan_for(cfg, corpus)['corpora'][corpus]['files']
        present = []; missing = []
        for item in entries:
            p = Path(item['local']); p = p/Path(item['path']).name if p.is_dir() else p
            (present if p.is_file() and p.stat().st_size == item['bytes'] else missing).append(item['index'])
        result['raw'][corpus] = dict(size_ready=present, missing=missing, size_only_inventory=True,
            hash_admission='shard_worker verifies SHA before tokenization; size alone is not a PASS')
    for name in ['status', 'failure', 'kg_verified', 'dot_e2_verified', 'dclm_verified', 'fineweb_edu_verified', 'complete']:
        p = output/(name+'.json'); result[name] = read(p) if p.is_file() else None
    if (output/'prepare.log').is_file():
        with (output/'prepare.log').open('rb') as f:
            f.seek(max(0, f.seek(0, 2)-6000)); result['log_tail'] = f.read().decode(errors='replace')
    print('A2_DOT_DATA_HEALTH', json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['launch','work','health','download-command'])
    p.add_argument('--commit'); p.add_argument('--source-manifest'); p.add_argument('--batch', type=int)
    args = p.parse_args(); cfg = recipe()
    if args.action == 'launch': launch(cfg, args.commit or '', args.source_manifest)
    elif args.action == 'health': health(cfg)
    elif args.action == 'download-command': print(download_command(cfg, args.batch), end='')
    else:
        try: work(cfg)
        except Exception as e:
            write_json(Path(cfg['output'])/'failure.json', dict(error=repr(e), utc=time.time(), preserved_partial_outputs=True))
            raise


if __name__ == '__main__':
    main()
