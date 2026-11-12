"""Additive E1-SR route/re-entry correction, without optimizer or training."""
import argparse
import json
import tarfile
from pathlib import Path

import numpy as np

from efficient_llm.dot_pool.data import sha, write
from scripts.dot_pool_evaluate import load, routes, e1_analysis


def check_inputs(expected):
    for name, record in expected['inputs'].items():
        path = Path(name)
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('Input identity differs: ' + name)
    manifest = Path(expected['data_manifest'])
    if sha(manifest) != expected['data_manifest_sha256']:
        raise ValueError('E1 data manifest changed')
    for name, record in json.loads(manifest.read_text())['files'].items():
        if Path(name).name != name:
            raise ValueError('Invalid data member path')
        path = manifest.parent / name
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('E1 data member changed: ' + name)


def verify_accuracy(final, archive):
    actual = {}
    with np.load(archive, allow_pickle=False) as values:
        for split in ['id_dev', 'ood_dev', 'id_test', 'ood_test']:
            correct = values[split + '_correct']
            target = final['metrics'][split]
            if len(correct) != target['targets'] or float(correct.mean()) != target['accuracy']:
                raise ValueError('Corrected SR accuracy differs from final evaluation: ' + split)
            actual[split] = {'targets': len(correct), 'correct': int(correct.sum()),
                             'accuracy': float(correct.mean())}
    return actual


def run(expected, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Refuse to overwrite an existing diagnostic correction')
    check_inputs(expected)
    original = Path(expected['run'])
    cfg = json.loads((original / 'config.json').read_text())
    if cfg['study'] != 'E1' or cfg['arm'] != 'SR' or cfg['seed'] not in [0, 1]:
        raise ValueError('This correction is restricted to the two E1-SR runs')
    final = json.loads((original / 'final_eval.json').read_text())
    if final['step'] != cfg['steps'] or final['step'] != 154000:
        raise ValueError('Unexpected final checkpoint step')
    output.mkdir(parents=True)
    write(output / 'started.json', {'original_run': str(original), 'inputs': expected['inputs']})
    # load() uses the original full-precision final state, as the original eval did.
    _, loaded_cfg, model = load(original)
    if loaded_cfg != cfg:
        raise ValueError('Configuration changed while loading')
    routes(output, cfg, model)
    actual = verify_accuracy(final, output / 'routes_sample.npz')
    e1_analysis(output)
    check_inputs(expected)
    proof = {'verified': True, 'run': original.name, 'original_run': str(original),
             'inputs_unchanged': True, 'training_performed': False,
             'seed_policy': 'seed9182 reset separately per E1 split; batch32, single rank',
             'split_accuracy': actual,
             'reentry': json.loads((output / 'reentry.json').read_text()),
             'files': {name: {'bytes': (output / name).stat().st_size,
                              'sha256': sha(output / name)}
                       for name in ['routes_sample.npz', 'reentry.json']}}
    write(output / 'verification.json', proof)
    archive = output / 'SR_DIAGNOSTICS_SMALL.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name in ['routes_sample.npz', 'reentry.json', 'verification.json']:
            tar.add(output / name, arcname=name, recursive=False)
    if archive.stat().st_size >= 8 * 1024 * 1024:
        raise ValueError('Correction exceeds small export cap')
    write(output / 'complete.json', {'verified': True, 'training_performed': False,
                                    'path': str(archive), 'bytes': archive.stat().st_size,
                                    'sha256': sha(archive)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.expected).read_text()), args.output)
