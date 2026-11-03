"""Offline, CPU-only reconstruction from the archived A2 data recipe.

Fresh launch only. Preserve backup and partial failures; publish success only
when historical content hashes (not timing-dependent manifest bytes) match.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys

from scripts.a2_prepare_corpora import sha, write_json, verify_packed

RECIPE = 'resources/a2_data_rebuild_wc69_20261103.json'
OUTPUT = Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-data-rebuild-20261103-a01')
DATA = Path('/mnt/local/_data/deep-llms_th2/a2_dclm_rebuild_20261103')
SHUFFLE = Path('/mnt/local/_data/deep-llms_th2/a2_dclm_shuffle2030_rebuild_20261103')
SESSION = 'a2_data_rebuild_wc69_20261103_a01'


def require(value, message):
    if not value:
        raise ValueError(message)


def compare_content(actual, expected, keys):
    for key in keys:
        require(actual[key] == expected[key], 'Historical content differs: ' + key)


def copy_verified(source, destination, expected):
    source, destination = Path(source), Path(destination)
    require(not source.is_symlink() and source.is_file(), 'Missing backup file: ' + str(source))
    require(sha(source) == expected, 'Backup hash mismatch: ' + str(source))
    require(not destination.exists() and not destination.is_symlink(), 'Destination already exists: ' + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive destination creation; partial copies are retained on failure.
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst, 8 << 20)
    require(sha(destination) == expected, 'Restored hash mismatch: ' + str(destination))
    return dict(source=str(source), destination=str(destination), sha256=expected,
                bytes=destination.stat().st_size)


def restore(recipe):
    backup = Path('/mnt/local/_backup/deep-llms_th2')
    source = backup/'6y3o-a2-inputs-20261028-a01/run/inputs'
    target = Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs')
    require(sha(source/'data/manifest.json') == recipe['culturax_manifest_sha256'], 'CulturaX manifest identity')
    require(sha(source/'assets.json') == recipe['assets_sha256'], 'Assets manifest identity')
    manifest = json.loads((source/'data/manifest.json').read_text())
    verify_packed(source/'data', manifest)
    for name, key in [('documents.jsonl', 'documents_sha256'), ('indices.npz', 'indices_sha256')]:
        require(sha(source/'data'/name) == manifest[key], 'CulturaX ' + name)
    require(not target.exists(), 'Refuse existing restored inputs')
    files = []
    for path in sorted(source.rglob('*')):
        require(not path.is_symlink(), 'Backup symlink refused')
        if path.is_file():
            files.append(copy_verified(path, target/path.relative_to(source), sha(path)))
    plan = json.loads(Path('resources/a2_inputs_20261028.json').read_text())
    expected_files = {}
    for name in ('dclm/P300', 'dclm/S72'):
        asset = plan['assets'][name]
        for directory, hashes in [(asset['weights_dir'], asset['files']), (asset['code_dir'], asset['code_files'])]:
            for filename, digest in hashes.items():
                expected_files[str(Path(directory)/filename)] = digest
    for filename, digest in plan['tokenizer_provenance']['gpt_neox']['files'].items():
        expected_files[str(Path(plan['tokenizer_path'])/filename)] = digest
    model_root = Path('/mnt/local/_models/deep-llms_th2')
    for destination, digest in expected_files.items():
        files.append(copy_verified(backup/Path(destination).relative_to(model_root), destination, digest))
    receipt = dict(status='verified', spec_sha256=recipe['spec_sha256'], files=files,
                   culturax_manifest_sha256=sha(target/'data/manifest.json'),
                   assets_sha256=sha(target/'assets.json'), backup_modified=False)
    write_json(OUTPUT/'restore.json', receipt)
    print('RESTORE_VERIFIED', json.dumps(receipt), flush=True)


def work():
    recipe = json.loads(Path(RECIPE).read_text())
    require(sha(recipe['original_plan']) == recipe['original_plan_sha256'], 'Original corpus plan changed')
    for package, expected in [('tokenizers', '0.22.1'), ('pyarrow', '26.0.0'), ('numpy', '2.4.6')]:
        require(version(package) == expected, 'Runtime pin mismatch: ' + package + '=' + version(package))
    restore(recipe)
    plan = json.loads(Path(recipe['original_plan']).read_text())
    plan['root'] = str(DATA)
    plan['corpora'] = {'dclm': plan['corpora']['dclm']}
    plan['corpora']['dclm']['files'] = plan['corpora']['dclm']['files'][:216]
    require([f['index'] for f in plan['corpora']['dclm']['files']] == recipe['selected_dclm_indices'], 'Source prefix selection')
    plan_path = OUTPUT/'plan.json'
    write_json(plan_path, plan)
    print('CORPUS_START', str(plan_path), flush=True)
    subprocess.run([sys.executable, '-u', '-m', 'scripts.a2_prepare_corpora', '--plan', str(plan_path), '--deadline-hours', '168'], check=True)
    packed = DATA/'packed/dclm'
    rebuilt = json.loads((packed/'manifest.json').read_text())
    expected = recipe['corpus_expected']
    compare_content(rebuilt, expected, ['streams', 'tokenizer_sha256', 'eos_id', 'small_indices', 'partition', 'packing'])
    prefix = lambda m: [(p['group'], p['source_index'], p['tokens']) for p in m['source_prefixes']]
    require(prefix(rebuilt) == prefix(expected), 'Historical per-stream source prefixes differ')
    write_json(OUTPUT/'corpus_identity.json', dict(status='verified', spec_sha256=recipe['spec_sha256'],
        old_manifest_sha256=recipe['shuffle_expected']['source_manifest_sha256'],
        new_manifest_sha256=sha(packed/'manifest.json'), content_hashes_equal=True, source_prefixes_equal=True))
    subprocess.run([sys.executable, '-u', '-m', 'scripts.a2_stage1_data', '--plan', str(plan_path),
        '--source-sha256', sha(packed/'manifest.json'), '--output', str(SHUFFLE), '--spec', 'docs/A2_stage1_spec.md'], check=True)
    shuffled = json.loads((SHUFFLE/'manifest.json').read_text())
    compare_content(shuffled, recipe['shuffle_expected'], ['streams', 'tokenizer_sha256', 'eos_id', 'order',
        'excluded_original_tail_tokens', 'dropped_after_shuffle', 'document_count', 'documents_sha256',
        'document_order_sha256', 'heldout_documents_sha256', 'document_dtype', 'T7', 'T18'])
    # Independently re-read the physical prefix and the production reader view.
    from efficient_llm.a2_depth.stream import hashes
    from efficient_llm.a2.data import PackedRows
    import numpy as np
    dt6 = recipe['historical_dt6']
    tokens = hashes(SHUFFLE/'heal.bin', 781248*4096*4)
    boundaries = hashes(SHUFFLE/'heal.starts', 781248*4096)
    require(tokens == dt6['tokens'] and boundaries == dt6['boundaries'], 'Historical full/prefix hashes differ')
    rows = PackedRows(SHUFFLE, 'heal', count=3124992, verify=False)
    require(len(rows) == 3124992 and np.array_equal(rows.order, np.arange(3124992)), 'Depth reader count/order')
    digest = hashlib.sha256()
    for begin in range(0, 781248, 4096):
        digest.update(rows.ids[rows.order[begin:min(begin+4096, 781248)]].astype('<u4', copy=False).tobytes(order='C'))
    require(digest.hexdigest() == dt6['reader_prefix_sha256'], 'Depth reader prefix mismatch')
    receipt = dict(status='verified', spec_sha256=recipe['spec_sha256'], utc=datetime.now(timezone.utc).isoformat(),
        original_manifest_sha256=dt6['historical_manifest_sha256'], new_manifest_sha256=sha(SHUFFLE/'manifest.json'),
        full_and_prefix_content_equal=True, document_order_equal=True, selected_sequences=3124992,
        prefix_sequences=781248, reader_prefix_sha256=digest.hexdigest(), tokens=tokens, boundaries=boundaries,
        corpus_manifest_sha256=sha(packed/'manifest.json'), restore_receipt_sha256=sha(OUTPUT/'restore.json'),
        source_manifest_sha256=sha(OUTPUT/'source-manifest.json'), stream=str(SHUFFLE),
        training_launched=False, watcher='OFF', note='Fresh provenance; historical receipts retained unchanged. Training admission plan still required.')
    write_json(OUTPUT/'complete.json', receipt)
    print('REBUILD_COMPLETE', json.dumps(receipt), flush=True)


def launch():
    require(socket.gethostname() == 'thiennh-p6-wc69-worker-0', 'Wrong node')
    for path in (OUTPUT, DATA, SHUFFLE):
        require(not path.exists(), 'Fresh output required: ' + str(path))
    require(subprocess.run(['tmux', 'has-session', '-t', SESSION], capture_output=True).returncode != 0, 'Session already exists')
    require(shutil.disk_usage('/mnt/local').free > 1_000_000_000_000, 'Need 1TB free')
    source = Path.cwd()
    frozen = OUTPUT/'source'
    names = subprocess.check_output(['git', 'ls-files'], text=True).splitlines()
    names = [n for n in names if n.startswith('efficient_llm/') and n.endswith('.py')] + [
        'scripts/a2_rebuild_wc69_data.py', 'scripts/a2_prepare_corpora.py', 'scripts/a2_stage1_data.py', 'scripts/a2_quant_io.py',
        RECIPE, 'resources/a2_corpora_30b_20261029.json', 'resources/a2_inputs_20261028.json', 'docs/A2_stage1_spec.md']
    identities = {}
    for name in names:
        target = frozen/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, target)
        identities[name] = sha(target)
    write_json(OUTPUT/'source-manifest.json', dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        files=identities, cpu_only=True, runtime={k: version(k) for k in ('numpy','pyarrow','tokenizers')}))
    env = ['env','CUDA_VISIBLE_DEVICES=','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','HF_DATASETS_OFFLINE=1',
           'HF_HUB_DISABLE_TELEMETRY=1','PYTHONDONTWRITEBYTECODE=1','TOKENIZERS_PARALLELISM=false',
           'RAYON_NUM_THREADS=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONPATH='+str(frozen)]
    command = 'exec ' + shlex.join(env+['nice','-n','10',sys.executable,'-u','-m','scripts.a2_rebuild_wc69_data','--work'])
    command += ' >' + shlex.quote(str(OUTPUT/'prepare.log')) + ' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',SESSION,'-c',str(frozen),command],check=True)
    print('CPU_REBUILD_SUBMITTED', json.dumps(dict(output=str(OUTPUT), session=SESSION, host=socket.gethostname())), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', action='store_true')
    args = parser.parse_args()
    if args.work:
        try:
            work()
        except Exception as error:
            write_json(OUTPUT/'failure.json', dict(error=type(error).__name__, message=str(error),
                utc=datetime.now(timezone.utc).isoformat(), spec_sha256=json.loads(Path(RECIPE).read_text())['spec_sha256']))
            raise
    else:
        launch()
