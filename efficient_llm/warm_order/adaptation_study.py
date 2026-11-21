"""CPU paired-snapshot study: matched loss, geometry and replay, no training.

Consumes existing pinned snapshots one at a time. Raw moments stay node-side.
No launch, remote access, automatic retry, GPU admission or freshness claim.
"""
import json
from pathlib import Path

import numpy as np

from .adaptation_pipeline import load_snapshot_panel, evaluate_stream
from .adaptation_timecourse import summarize
from .adaptation_geometry_pipeline import COMPARISONS, evaluate_geometry, replay_geometry
from .data import sha
from .horizon_train import write_json
from .adaptation_runtime import measurement_runtime, measurement_device


def _read(path):
    return json.loads(Path(path).read_text())


def join(snapshot_directories):
    """Replay all panels, then bind geometry to the exact loss checkpoint/rows."""
    directories = list(map(Path, snapshot_directories))
    trajectory = summarize([d/'loss' for d in directories])
    for source, directory in zip(trajectory['sources'], directories):
        source['directory'] = directory.name+'/loss'
    signatures = {}
    for point, directory in zip(trajectory['points'], directories):
        loss_manifest = _read(directory/'loss/manifest.json')
        loss_provenance = loss_manifest['provenance']
        with np.load(directory/'loss/scores.npz', allow_pickle=False) as raw:
            loss_ids = raw['row_ids'].copy()
        geometry = {}
        for comparison in COMPARISONS:
            root = directory/'geometry'/comparison
            replay_geometry(root)
            manifest = _read(root/'manifest.json')
            if (manifest['comparison'] != comparison or
                    manifest['provenance'] != loss_provenance['checkpoints'] or
                    manifest['verified_input_file_hashes'] != loss_provenance['verified_input_file_hashes'] or
                    manifest['config'] != loss_manifest['config'] or
                    manifest.get('measurement_runtime') != loss_manifest.get('measurement_runtime') or
                    not np.array_equal(np.load(root/'row_ids.npy', allow_pickle=False), loss_ids)):
                raise ValueError('Geometry and loss checkpoint/data/row pairing mismatch')
            signature = {key: manifest[key] for key in ('keys', 'positions', 'labels', 'batch_size', 'measurement_runtime')}
            if comparison not in signatures: signatures[comparison] = signature
            elif signatures[comparison] != signature:
                raise ValueError('Geometry selection/alignment changed across snapshots')
            geometry[comparison] = dict(metrics=_read(root/'REPORT.json'),
                                       manifest_sha256=sha(root/'manifest.json'))
        point['geometry'] = geometry
    trajectory['geometry_selection'] = signatures
    trajectory['interpretation'] = ('Matched loss and descriptive geometry trajectories; '
        'geometry changes do not identify causes. Inspect registered component and weight interventions separately.')
    return trajectory


def _validate(config):
    if config['schema'] != 1 or config['domain'] != 'dclm': raise ValueError('DCLM study schema required')
    snapshots = config['snapshots']
    steps = [s['step'] for s in snapshots]
    if (len(steps) < 2 or any(type(s) is not int or s < 0 for s in steps) or
            steps[0] != 0 or steps != sorted(set(steps))):
        raise ValueError('Step zero and increasing unique snapshots required')
    for snapshot in snapshots:
        if set(snapshot) != {'step', 'phase_files'} or set(snapshot['phase_files']) != {'A', 'B'}:
            raise ValueError('Both pinned phase snapshots required')
        for spec in snapshot['phase_files'].values():
            if set(spec) != {'path', 'sha256'} or sha(spec['path']) != spec['sha256']:
                raise ValueError('Study checkpoint hash mismatch')
    inputs = {p: sha(p) for p in (config['data']+'.bin', config['data']+'.starts', config['indices'])}
    if inputs != config['expected_inputs']: raise ValueError('Study input hash mismatch')


def run(config, out):
    with measurement_runtime(config.get('measurement_device','cpu')):
        return _run(config,out)


def _run(config, out):
    """Fresh output only; interrupted studies retain partial evidence, no relaunch."""
    out = Path(out)
    if out.exists() and any(out.iterdir()): raise ValueError('Fresh study output required')
    _validate(config)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'config.json', config)
    directories, files = [], {'config.json': sha(out/'config.json')}
    for snapshot in config['snapshots']:
        models, proof = load_snapshot_panel(config['parent'], config['parent_hashes'], snapshot['phase_files'])
        if proof['phase_step'] != snapshot['step']: raise ValueError('Declared snapshot step mismatch')
        for model in models.values(): model.to(measurement_device())
        directory = out/f"step_{snapshot['step']:06d}"; directories.append(directory)
        evaluate_stream(models, config['data'], config['indices'], config['expected_inputs'],
                        config['batch_size'], directory/'loss', proof)
        for comparison in COMPARISONS:
            evaluate_geometry(models, comparison, config['data'], config['indices'], config['expected_inputs'],
                config['batch_size'], tuple(config['positions']), tuple(config['geometry_keys']),
                directory/'geometry'/comparison, proof, config['max_moment_bytes'])
        # Keep all sufficient statistics locally/on-node; pin each panel manifest.
        for panel in [directory/'loss'] + [directory/'geometry'/c for c in COMPARISONS]:
            name = str((panel/'manifest.json').relative_to(out)); files[name] = sha(out/name)
        del models
    report = join(directories)
    write_json(out/'REPORT.json', report); files['REPORT.json'] = sha(out/'REPORT.json')
    write_json(out/'manifest.json', dict(schema=1, snapshots=[d.name for d in directories], files=files,
        production_ready=False, weights_exported=False, raw_moments_exportable=False,
        scope='Paired study assembly; no implicit node/GPU admission'))
    replay(out)
    return report


def replay(out):
    out = Path(out); manifest = _read(out/'manifest.json'); config = _read(out/'config.json')
    directories = [f"step_{s['step']:06d}" for s in config['snapshots']]
    if manifest['schema'] != 1 or manifest['snapshots'] != directories:
        raise ValueError('Study snapshot layout mismatch')
    expected = {'REPORT.json', 'config.json'}
    for directory in directories:
        expected.add(directory+'/loss/manifest.json')
        expected.update(directory+'/geometry/'+c+'/manifest.json' for c in COMPARISONS)
    if set(manifest['files']) != expected: raise ValueError('Study file set mismatch')
    for name, digest in manifest['files'].items():
        if sha(out/name) != digest: raise ValueError('Study file hash mismatch')
    # Bind assembly config to measured pins, not merely to relative directory names.
    for spec, directory in zip(config['snapshots'], directories):
        measured = _read(out/directory/'loss/manifest.json')
        p = measured['provenance']; ck = p['checkpoints']
        if (ck['phase_step'] != spec['step'] or ck['phase_files'] != spec['phase_files'] or
                ck['parent_file_pins'] != config['parent_hashes'] or
                p['verified_input_file_hashes'] != config['expected_inputs'] or
                p['batch_size'] != config['batch_size']):
            raise ValueError('Study registration mismatch')
        for comparison in COMPARISONS:
            g = _read(out/directory/'geometry'/comparison/'manifest.json')
            if g['keys'] != config['geometry_keys'] or g['positions'] != config['positions']:
                raise ValueError('Study geometry registration mismatch')
    report = join([out/d for d in directories])
    if report != _read(out/'REPORT.json'): raise ValueError('Study numeric replay mismatch')
    return dict(status='PASS', snapshots=len(directories), production_ready=False)
