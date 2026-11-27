"""Register historical dense-parent pins from an authenticated local export.

Metadata only. No checkpoint load, remote request, fresh availability claim or
selection of new continuation/evaluation rows.
"""
import hashlib
import json
import tarfile
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8<<20), b''): h.update(block)
    return h.hexdigest()


def register(root):
    root = Path(root)
    study = root/'results/horizon_20261119'
    inventory_path = root/'results/layer_dependency_20261120/horizon_row_inventory.json'
    inventory = json.loads(inventory_path.read_text())
    backup_path = study/'backup_ledger.json'; backup = json.loads(backup_path.read_text())
    archive = study/'export_a01/results.tar.gz'
    archive_sha = digest(archive)
    if archive_sha != inventory['archive_sha256'] or archive_sha != backup['local_replay']['archive_sha256']:
        raise ValueError('Historical archive identity mismatch')
    base = study/'export_a01/verified'
    member_manifest = json.loads((base/'export-members.json').read_text())
    inputs, parents = {}, {}
    with tarfile.open(archive, 'r:gz') as tar:
        names = tar.getnames()
        if len(names) != len(set(names)): raise ValueError('Duplicate archive members')
        def read(name):
            member = tar.getmember(name)
            if not member.isfile() or member.size > 2_000_000: raise ValueError('Unexpected metadata member')
            data = tar.extractfile(member).read()
            h = hashlib.sha256(data).hexdigest()
            if h != member_manifest[name] or data != (base/name).read_bytes():
                raise ValueError('Extracted member/hash mismatch')
            if name in inventory['verified_horizon_members'] and h != inventory['verified_horizon_members'][name]:
                raise ValueError('Historical inventory mismatch')
            inputs[name] = h
            return data
        for seed in (0, 1):
            prefix = f'runs/D-s{seed}/train/'
            done = json.loads(read(prefix+'complete.json'))
            cp = json.loads(read(prefix+'checkpoint.json'))
            recipe = json.loads(read(prefix+'recipe.json'))
            trace = [json.loads(x) for x in read(prefix+'validation.jsonl').splitlines()]
            c = recipe['config']
            if done['arm'] != 'D' or done['seed'] != seed or done['step'] != 80000 or c['steps'] != 80000:
                raise ValueError('Parent identity mismatch')
            if cp['sha256'] != done['fullstate_sha256'] or cp['step'] != 80000:
                raise ValueError('Checkpoint metadata mismatch')
            if [r['step'] for r in trace] != list(range(0, 80001, 5000)):
                raise ValueError('Validation trajectory incomplete')
            for path, h in recipe['source'].items():
                if digest(root/path) != h: raise ValueError('Parent implementation differs: '+path)
            pins = {'fullstate.pt': done['fullstate_sha256'], 'weights.pt': done['weights_sha256'],
                    **{f: inputs[prefix+f] for f in ('recipe.json', 'complete.json', 'checkpoint.json')}}
            copies = {path: h for path, h in backup['node_copies'].items()
                      if path.endswith('/'+prefix+'fullstate.pt') or path.endswith('/'+prefix+'weights.pt')}
            if len(copies) != 2 or set(copies.values()) != {pins['fullstate.pt'], pins['weights.pt']}:
                raise ValueError('Historical backup pins disagree')
            tail = [{'step': r['step'], 'canonical_validation_nll': r['full_order_means'][0]} for r in trace[-4:]]
            parents[f'D-s{seed}'] = dict(file_pins=pins, model=c['model'], batch=c['batch'],
                terminal_lr=.1*c['lr'], original_parameter_order_source=recipe['source'],
                historical_backup_copies=copies, node_availability='not freshly inspected',
                fullstate_bytes_verified_locally=False, training_inputs=recipe['inputs'],
                validation_tail=tail, validation_change_last10000=tail[-1]['canonical_validation_nll']-tail[-3]['canonical_validation_nll'],
                maturity='Still improving on reused128-row monitor; convergence not established')
    return dict(status='historical_parent_registration_only', archive_sha256=archive_sha,
                evidence_pins={str(inventory_path.relative_to(root)): digest(inventory_path),
                               str(backup_path.relative_to(root)): digest(backup_path)},
                verified_metadata_members=inputs, parents=parents,
                continuation_rows_registered=False, confirmation_rows_registered=False,
                current_node_ownership_verified=False, production_ready=False,
                remote_requests=0)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    result = register(root)
    out = root/'results/layer_dependency_20261120/adaptation_parent_registration.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: {'lr': v['terminal_lr'], 'last10000_delta': v['validation_change_last10000']}
                      for k, v in result['parents'].items()}))
