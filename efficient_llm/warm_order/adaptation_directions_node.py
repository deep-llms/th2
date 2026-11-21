"""CPU-calibrated, parity-gated direction evaluation. Ownership is external."""
import argparse
import shutil
import tarfile
from pathlib import Path

import torch

from .adaptation_pipeline import load_snapshot_panel
from .adaptation_factors import factors_crossovers
from .adaptation_directions_pipeline import (read, validate, selections, check_proof, calibrate,
    replay_calibration, panel, evaluate_stream, replay)
from .adaptation_runtime import measurement_runtime, measurement_policy
from .data import Rows, sha
from .horizon_train import write_json


def load(c):
    validate(c)
    models, proof = load_snapshot_panel(c['parent'], c['parent_hashes'], c['phase_files'])
    check_proof(c, proof)
    return models, proof


@torch.no_grad()
def reference(models, x, fitted):
    ref, _ = panel(models, x, fitted, reference=True)
    policy = measurement_policy()
    for b in ('A', 'B'):
        prior = factors_crossovers(models['parent'], models[b], x)
        for cell, endpoint in [('base', 'attn/000'), ('full', 'attn/001')]:
            torch.testing.assert_close(ref[b+'/'+cell]['hidden'], prior['cells'][endpoint].cpu(),
                atol=policy['component_atol'], rtol=policy['component_rtol'])
    return ref


def prepare(plan, expected_sha, out):
    plan, out = Path(plan), Path(out)
    if sha(plan) != expected_sha or out.exists():
        raise ValueError('Plan hash/fresh output required')
    configs = read(plan)['seeds']
    if set(configs) != {'0', '1'}:
        raise ValueError('Two seed configurations required')
    out.mkdir(parents=True); receipts = {}
    for seed, c in configs.items():
        models, proof = load(c); rows, cal, _ = selections(c, models['parent'].c.length)
        root = out/f's{seed}'; root.mkdir()
        with measurement_runtime('cpu'):
            calibrate(models, c, proof, root/'calibration')
            fitted = replay_calibration(root/'calibration')
            ref = reference(models, rows.batch(cal[:1], 'cpu'), fitted)
        torch.save(ref, root/'cpu-reference.pt')
        c = dict(c, calibration_row=int(cal[0]), cpu_reference_sha256=sha(root/'cpu-reference.pt'),
                 calibration_artifact_sha256=sha(root/'calibration/calibration.json'))
        write_json(root/'config.json', c)
        receipts[seed] = dict(config_sha256=sha(root/'config.json'),
            cpu_reference_sha256=c['cpu_reference_sha256'], checkpoints=proof,
            calibration_artifact_sha256=c['calibration_artifact_sha256'], reference_cells=len(ref))
        del models, ref, fitted
    write_json(out/'prepared.json', dict(seeds=receipts, actual_panels_loaded=2, models_per_panel=3,
        cpu_native_parity=True, train_updates=0, registered_plan_sha256=expected_sha,
        basis_fitted_on_calibration_only=True, gpu_admitted=False))
    return read(out/'prepared.json')


def run_prepared(root, expected_sha, device='cuda'):
    root = Path(root)
    if sha(root/'config.json') != expected_sha:
        raise ValueError('Prepared config mismatch')
    c = read(root/'config.json'); validate(c)
    if c['measurement_device'] != device:
        raise ValueError('Runtime differs from registered config')
    if sha(root/'cpu-reference.pt') != c['cpu_reference_sha256']:
        raise ValueError('CPU reference mismatch')
    if sha(root/'calibration/calibration.json') != c['calibration_artifact_sha256']:
        raise ValueError('Calibration registration mismatch')
    if (root/'study').exists() or (root/'gate.json').exists():
        raise ValueError('Fresh seed output required')
    models, proof = load(c); rows, cal, _ = selections(c, models['parent'].c.length)
    if int(cal[0]) != c['calibration_row']:
        raise ValueError('Calibration reference row changed')
    fitted = replay_calibration(root/'calibration')
    saved = torch.load(root/'cpu-reference.pt', map_location='cpu', weights_only=True)
    with measurement_runtime(device) as policy:
        for model in models.values():
            model.to(device)
        current = reference(models, rows.batch(cal[:1], device), fitted); maxima = {}
        if set(current) != set(saved):
            raise ValueError('Reference cells differ')
        for key, fields in current.items():
            if set(fields) != set(saved[key]):
                raise ValueError('Reference fields differ')
            maxima[key] = {}
            for name, value in fields.items():
                torch.testing.assert_close(value, saved[key][name],
                    atol=policy['component_atol'], rtol=policy['component_rtol'])
                maxima[key][name] = float((value-saved[key][name]).abs().max())
        write_json(root/'gate.json', dict(pass_=True, policy=policy, max_abs=maxima,
            config_sha256=expected_sha, calibration_artifact_sha256=c['calibration_artifact_sha256']))
        del saved, current
        result = evaluate_stream(models, c, proof, root/'calibration', root/'study')
        write_json(root/'complete.json', replay(root/'study'))
    return dict(cases=len(result['cases']), sequences=result['sequences'], gate=True)


def finish(base):
    base = Path(base); prepared = read(base/'prepared/prepared.json'); files = []; reports = {}; copies = {}
    backup = base/'basis-backup'
    if backup.exists():
        raise ValueError('Fresh node backup required')
    # Authenticate/replay both panels before making an export.
    for seed in ('0', '1'):
        root = base/'prepared'/f's{seed}'; study = root/'study'
        replay(study); replay_calibration(root/'calibration')
        c = read(root/'config.json'); gate = read(root/'gate.json')
        if (sha(root/'config.json') != prepared['seeds'][seed]['config_sha256'] or
                c != read(study/'config.json') or gate['pass_'] is not True or
                gate['config_sha256'] != sha(root/'config.json') or
                gate['calibration_artifact_sha256'] != c['calibration_artifact_sha256'] or
                sha(root/'calibration/calibration.json') != c['calibration_artifact_sha256'] or
                read(root/'calibration/calibration.json') != read(study/'calibration.json')):
            raise ValueError('Prepared/gate/calibration/study mismatch')
        reports[seed] = read(study/'REPORT.json')
        files += [root/'config.json', root/'gate.json', root/'complete.json']
        files += [study/n for n in ('config.json', 'manifest.json', 'REPORT.json', 'scores.npz',
                                    'diagnostics.npz', 'calibration.json')]
    for seed in ('0', '1'):
        source = base/'prepared'/f's{seed}'/'calibration'; target = backup/f's{seed}'
        shutil.copytree(source, target)
        replay_calibration(target)
        for name, h in read(source/'calibration.json')['node_files'].items():
            if sha(target/name) != h:
                raise ValueError('Node basis backup mismatch')
            copies[str((target/name).relative_to(base))] = h
    write_json(base/'REPORT.json', dict(seeds=reports,
        scope='Frozen V/O direction diagnostic; adapted donor required per input; exploratory, not causal shares'))
    files += [base/'REPORT.json', base/'prepared/prepared.json', base/'execution/source.json', base/'execution/admission.json']
    inventory = {str(p.relative_to(base)): sha(p) for p in files}
    write_json(base/'small-files.json', inventory)
    with tarfile.open(base/'tier1.tar.gz', 'w:gz') as tar:
        for p in sorted(files+[base/'small-files.json']):
            if p.is_symlink() or p.suffix == '.pt' or 'moments' in p.name or 'basis' in p.name:
                raise ValueError('Forbidden export member')
            tar.add(p, arcname=str(p.relative_to(base)), recursive=False)
    if (base/'tier1.tar.gz').stat().st_size > 5_000_000:
        raise ValueError('Archive too large')
    proof = dict(archive_sha256=sha(base/'tier1.tar.gz'), archive_bytes=(base/'tier1.tar.gz').stat().st_size,
        files=inventory, node_replay=True, node_basis_moment_backups=copies,
        weights_exported=False, reference_activations_exported=False, basis_exported=False, moments_exported=False,
        checkpoint_policy='Existing fullstates/32priorbackups unchanged; no train updates')
    write_json(base/'backup.json', proof)
    return proof


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['prepare', 'run'])
    p.add_argument('--path', required=True); p.add_argument('--sha', required=True)
    p.add_argument('--out'); p.add_argument('--device', default='cuda'); a = p.parse_args()
    torch.set_num_threads(4)
    result = prepare(a.path, a.sha, a.out) if a.mode == 'prepare' else run_prepared(a.path, a.sha, a.device)
    print('DIRECTIONS_'+a.mode.upper()+'_COMPLETE', result, flush=True)


if __name__ == '__main__':
    main()
