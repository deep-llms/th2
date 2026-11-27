"""Read-only local Horizon exposure inventory; does not certify document disjointness.

Authenticate recorded recipes and row-ID arrays, without opening node paths,
loading weights, fitting a model, or modifying historical exports. Emit JSON
on stdout only. This is one historical family, not the complete exclusion union.
"""
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def describe(ids):
    a = np.asarray(ids)
    if a.ndim != 1 or a.dtype.kind not in 'iu' or not len(a) or (a < 0).any():
        raise ValueError('Invalid row IDs')
    if len(np.unique(a)) != len(a):
        raise ValueError('Duplicate row IDs')
    return dict(rows=len(a), minimum=int(a.min()), maximum=int(a.max()),
                ordered_int64le_sha256=hashlib.sha256(a.astype('<i8').tobytes()).hexdigest())


def audit():
    base = ROOT / 'results/horizon_20261119'
    export = base / 'export_a01/verified'
    handoff_path = base / 'node_handoff_verified.json'
    handoff = json.loads(handoff_path.read_text())
    archive = base / 'export_a01/results.tar.gz'
    if sha(archive) != handoff['archive_sha256']:
        raise ValueError('Horizon archive hash mismatch')
    with tarfile.open(archive, 'r:gz') as tar:
        matches = [m for m in tar if m.name.removeprefix('./') == 'export-members.json']
        if len(matches) != 1 or not matches[0].isfile():
            raise ValueError('Ambiguous manifest')
        manifest = json.load(tar.extractfile(matches[0]))
    verified = {}

    def member(name, arrays=False):
        path = export / name
        digest = sha(path)
        if digest != manifest[name]:
            raise ValueError('Member mismatch: ' + name)
        verified[name] = digest
        if arrays:
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files if k == 'ids'}
        return json.loads(path.read_text())

    prepared = member('inputs/complete.json')
    eval_path = ROOT / 'results/adaptive_refinement_20261119/export_a01/verified/inputs/eval.npz'
    eval_sha = sha(eval_path)
    with np.load(eval_path, allow_pickle=False) as z:
        ids = {k: z[k].copy() for k in ('fit', 'valid', 'dclm', 'fwe')}
    descriptions = {k: describe(v) for k, v in ids.items()}
    # This checks row IDs only, not document identities.
    if np.intersect1d(ids['fit'], ids['valid']).size:
        raise ValueError('Router row overlap')
    runs = []
    train_refs = {}
    first_eval = None
    source_verified = {}
    source_manifest = member('execution/source.json')
    for rel in ('efficient_llm/warm_order/horizon_prepare.py',
                'efficient_llm/warm_order/horizon_eval.py',
                'efficient_llm/warm_order/horizon_train.py',
                'efficient_llm/warm_order/horizon_probe.py'):
        digest = sha(ROOT / rel)
        if source_manifest[rel] != digest:
            raise ValueError('Execution source differs: ' + rel)
        source_verified[rel] = digest
    for arm in ('D', 'R', 'WS', 'WL'):
        for seed in (0, 1):
            name = f'{arm}-s{seed}'
            prefix = f'runs/{name}'
            cfg = member(prefix + '/config.json')
            recipe = member(prefix + '/train/recipe.json')
            complete = member(prefix + '/train/complete.json')
            if cfg['train'] != recipe['config'] or complete['step'] != cfg['train']['steps']:
                raise ValueError('Recipe/terminal mismatch')
            if prepared['files'][name + '.json'] != verified[prefix + '/config.json']:
                raise ValueError('Prepared config mismatch')
            ev = cfg['eval']
            if first_eval is None:
                first_eval = ev
            if ev != first_eval or cfg['hashes'][ev['indices']] != eval_sha:
                raise ValueError('Evaluation reservation mismatch')
            tc = cfg['train']
            if tc['probe_indices'] != ev['indices'] or tc['data'] != ev['stems']['fit'] or tc['model']['length'] != 512:
                raise ValueError('Unexpected stream/probe semantics')
            if recipe['inputs'][ev['indices']] != eval_sha:
                raise ValueError('Actual probe input mismatch')
            # Pin the actual training/probe source used to interpret array slicing.
            for rel in ('efficient_llm/warm_order/horizon_train.py',
                        'efficient_llm/warm_order/horizon_probe.py'):
                digest = sha(ROOT / rel)
                if recipe['source'][rel] != digest:
                    raise ValueError('Historical source differs: ' + rel)
                source_verified[rel] = digest
            rowpath = tc['indices']
            rowsha = prepared['files'][Path(rowpath).name]
            if recipe['inputs'][rowpath] != rowsha or tc['expected_inputs'][rowpath] != rowsha:
                raise ValueError('Training row pin mismatch')
            if rowpath in train_refs and train_refs[rowpath]['sha256'] != rowsha:
                raise ValueError('Conflicting row pin')
            train_refs[rowpath] = dict(sha256=rowsha, consumed_prefix_rows=tc['steps']*tc['batch'],
                                      locally_verified_values=False,
                                      required_mapping='E2 destination r -> original packed-heal order[r]')
            valid = member(prefix + '/validation.npz', arrays=True)['ids']
            if not np.array_equal(valid, ids['valid']):
                raise ValueError('Actual validation rows differ')
            for domain in ('dclm', 'fwe'):
                actual = member(prefix + f'/{domain}/summary.npz', arrays=True)['ids']
                if not np.array_equal(actual, ids[domain]):
                    raise ValueError('Actual final rows differ')
            runs.append(dict(name=name, training_rows=rowpath,
                             router_fit='eval.npz:fit', router_validation='eval.npz:valid',
                             monitor_rows=describe(ids['valid'][:tc['probe_size']]),
                             final_rows_match_registered=True))
    return dict(status='LOCAL_ROW_INVENTORY_ONLY', production_ready=False,
        whole_document_mapping_verified=False, full_historical_union_complete=False,
        scope='All eight Horizon runs only; does not certify other experiment histories',
        audit_source_sha256=sha(Path(__file__)),
        archive_sha256=sha(archive), handoff_sha256=sha(handoff_path),
        verified_horizon_members=verified, actual_source_verified=source_verified,
        eval_array=dict(local_path=str(eval_path.relative_to(ROOT)), sha256=eval_sha,
                        node_path=first_eval['indices'], arrays=descriptions,
                        stems=first_eval['stems']),
        training_array_dependencies=train_refs, runs=runs,
        missing=['Actual values of both training row arrays',
                 'Pinned original corpus document indices/starts and E2 order',
                 'Other experiment training/router/validation/final/diagnostic exposures',
                 'Cross-corpus whole-document exclusion union and locked reservation'],
        limitations=['Row-ID equality is not source-document independence.',
                     'Local files and receipts do not establish current node availability.',
                     'No weights loaded, no inference/training, no remote access.'])


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2, allow_nan=False))
