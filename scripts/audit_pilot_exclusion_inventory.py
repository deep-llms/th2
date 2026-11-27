"""Offline pilot row-exposure audit. No model, remote call, or output mutation.

Reads the original compact archive and local Git history. The JSON on stdout
is a row inventory, never a certificate of original-document independence.
"""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

import numpy as np

from scripts.audit_horizon_exclusion_inventory import audit as horizon_audit
from scripts.audit_horizon_exclusion_inventory import describe, sha, ROOT


def read_export(base):
    ledger = json.loads((base / 'backup_ledger.json').read_text())
    proof_path = base / 'export_a01/proof.json'
    archive = base / 'export_a01/results.tar.gz'
    for path in (proof_path, archive):
        pin = ledger['local_files'][str(path.relative_to(base))]
        if sha(path) != pin['sha256'] or path.stat().st_size != pin['bytes']:
            raise ValueError('Historical ledger mismatch: ' + str(path))
    proof = json.loads(proof_path.read_text())
    if sha(archive) != proof['archive_sha256'] or archive.stat().st_size != proof['bytes']:
        raise ValueError('Archive proof mismatch')
    payloads = {}
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        if sum(m.size for m in members) > 100_000_000:
            raise ValueError('Oversized compact export')
        for m in members:
            p = PurePosixPath(m.name)
            if not m.isfile() or p.is_absolute() or '..' in p.parts or m.name in payloads:
                raise ValueError('Unsafe or duplicate archive member')
            b = tar.extractfile(m).read()
            if hashlib.sha256(b).hexdigest() != proof['files'].get(m.name):
                raise ValueError('Archive member mismatch')
            payloads[m.name] = b
    if payloads.keys() != proof['files'].keys():
        raise ValueError('Incomplete export')
    return payloads, proof


def array_ids(payload, key):
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        ids = z[key].copy()
    describe(ids)  # Reject malformed, nonintegral and duplicate selections.
    return ids


def compare_rows(left, right):
    # Numeric IDs from unrelated packed streams must never be joined.
    if left['stream'] != right['stream']:
        return dict(status='different_stream_not_compared')
    if left['stream_hashes'] != right['stream_hashes']:
        raise ValueError('Conflicting versions of the same stream')
    return dict(status='row_comparison_only',
                shared_rows=int(np.intersect1d(left['ids'], right['ids']).size))


def audit():
    base = ROOT / 'results/warm_order_20261118'
    payloads, proof = read_export(base)
    obj = lambda name: json.loads(payloads[name])
    prepared = obj('inputs/complete.json')
    sources = obj('execution/source.json')
    source_evidence = {}
    for rel in ('efficient_llm/warm_order/data.py', 'efficient_llm/warm_order/train.py',
                'efficient_llm/warm_order/prepare.py', 'efficient_llm/warm_order/evaluate.py',
                'resources/warm_order_data_20261118.json', 'scripts/warm_order_queue.py'):
        # The queue was subsequently modified. Read its exact archived revision;
        # never import or execute it. SHA equality, not the Git name, is the gate.
        b = (subprocess.run(['git', 'show', '73bd4303:' + rel], cwd=ROOT,
                            check=True, capture_output=True).stdout
             if rel == 'scripts/warm_order_queue.py' else (ROOT / rel).read_bytes())
        if hashlib.sha256(b).hexdigest() != sources[rel]:
            raise ValueError('Historical source mismatch: ' + rel)
        source_evidence[rel] = dict(sha256=sources[rel],
            origin='git:73bd4303' if rel == 'scripts/warm_order_queue.py' else 'current_local')
    plan = json.loads((ROOT / 'resources/warm_order_data_20261118.json').read_text())
    if plan['files'] != prepared['input_hashes']:
        raise ValueError('Data plan differs from prepared pins')
    exposures, train_refs, runs = {}, {}, []

    def exposure(name, ids, stem, pins):
        stream_hashes = {suffix: pins[stem + suffix] for suffix in ('.bin', '.starts')}
        exposures[name] = dict(ids=ids, stream=stem, stream_hashes=stream_hashes)

    for seed in (0, 1):
        for arm in ('D', 'R', 'W'):
            name = f'{arm}-s{seed}'
            recipe = obj(f'runs/{name}/recipe.json')
            cfg = recipe['config']
            done = obj(f'runs/{name}/complete.json')
            report = obj(f'eval/{name}/report.json')
            if cfg['arm'] != arm or cfg['seed'] != seed or done['step'] != cfg['steps']:
                raise ValueError('Training identity/terminal mismatch')
            if report['weights_sha256'] != done['weights_sha256']:
                raise ValueError('Wrong evaluation checkpoint')
            if cfg['data'] != plan['stems']['fit'] or cfg['model']['length'] != 512:
                raise ValueError('Unexpected training stream')
            if recipe['inputs'] != cfg['expected_inputs']:
                raise ValueError('Actual input pins differ')
            for rel, digest in recipe['source'].items():
                if sources[rel] != digest:
                    raise ValueError('Recipe source differs from execution')
            path = cfg['indices']
            pin = prepared['files'][Path(path).name]
            if recipe['inputs'][path] != pin:
                raise ValueError('Training row pin mismatch')
            for suffix in ('.bin', '.starts'):
                if recipe['inputs'][cfg['data'] + suffix] != plan['files'][cfg['data'] + suffix]:
                    raise ValueError('Training data pin mismatch')
            ref = dict(sha256=pin, consumed_rows=cfg['steps'] * cfg['batch'],
                       locally_verified_values=False)
            if path in train_refs and train_refs[path] != ref:
                raise ValueError('Conflicting training reference')
            train_refs[path] = ref
            ids = array_ids(payloads[f'eval/{name}/test_costs.npz'], 'ids')
            if len(ids) != 8192:
                raise ValueError('Unexpected pilot final coverage')
            final_key = 'pilot_final'
            if final_key in exposures and not np.array_equal(exposures[final_key]['ids'], ids):
                raise ValueError('Final IDs differ across runs')
            exposure(final_key, ids, plan['stems']['test'], plan['files'])
            if arm != 'D':
                b = payloads[f'eval/{name}/fit_dev_costs.npz']
                for key, role, count in (('fit_ids', 'fit', 8192), ('dev_ids', 'dev', 2048)):
                    ids = array_ids(b, key)
                    label = f'pilot_fit_s{seed}' if role == 'fit' else 'pilot_dev'
                    if len(ids) != count:
                        raise ValueError('Unexpected router coverage')
                    if label in exposures and not np.array_equal(exposures[label]['ids'], ids):
                        raise ValueError('Router row IDs differ across matched runs')
                    exposure(label, ids, plan['stems'][role], plan['files'])
            runs.append(dict(name=name, training_rows=path, router_fitted=arm != 'D',
                             final='pilot_final'))

    # Re-run the independent Horizon archive/source/actual-ID audit before joins.
    horizon = horizon_audit()
    hp = json.loads((ROOT / 'results/horizon_20261119/export_a01/verified/inputs/complete.json').read_text())
    with np.load(ROOT / horizon['eval_array']['local_path'], allow_pickle=False) as z:
        for key in ('fit', 'valid', 'dclm', 'fwe'):
            stem = horizon['eval_array']['stems']['fit' if key == 'valid' else key]
            exposure('horizon_' + key, z[key].copy(), stem, hp['hashes'])
    comparisons = {}
    for a in ('pilot_fit_s0', 'pilot_fit_s1', 'pilot_dev', 'pilot_final'):
        for b in ('horizon_fit', 'horizon_valid', 'horizon_dclm', 'horizon_fwe'):
            comparisons[a + '__' + b] = compare_rows(exposures[a], exposures[b])
    return dict(status='LOCAL_ROW_INVENTORY_ONLY', production_ready=False,
        whole_document_mapping_verified=False, full_historical_union_complete=False,
        audit_source_sha256=sha(Path(__file__)), archive_sha256=proof['archive_sha256'],
        verified_pilot_members=len(payloads), source_evidence=source_evidence,
        training_array_dependencies=train_refs, runs=runs,
        exposures={k: dict(stream=v['stream'], stream_hashes=v['stream_hashes'],
                           **describe(v['ids'])) for k, v in exposures.items()},
        row_comparisons=comparisons,
        horizon_archive_sha256=horizon['archive_sha256'],
        horizon_audit_source_sha256=horizon['audit_source_sha256'],
        limitations=['Observed IDs are recovered from actual evaluation exports; original pilot eval NPZ files are absent.',
                     'Training row values and original-document mapping remain unavailable.',
                     'Zero shared row IDs does not prove zero shared source documents.',
                     'Other historical experiments and cross-corpus document union remain incomplete.',
                     'No original results changed; no model, fitting, GPU, or remote operation.'])


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2, allow_nan=False))
