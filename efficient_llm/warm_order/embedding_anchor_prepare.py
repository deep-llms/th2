"""Hash-bound CPU initialization and numerical references; zero optimizer steps."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from .adaptation_runtime import measurement_runtime
from .attention_shift import fingerprint
from .data import sha
from .embedding_anchor import TRAIN_PATHS, HELD_OUT_PATHS
from .embedding_anchor_data import BoundRows
from .embedding_anchor_train import CPUStreamTrainer, sources
from .model import Config


REFERENCE_CASES = {
    'single0': dict(path=[0], alpha=1., donor=None),
    'single3': dict(path=[3], alpha=1., donor=None),
    'donor_start': dict(path=[1], alpha=.05, donor=0),
    'donor_end': dict(path=[1], alpha=.5, donor=0),
    'donor_direct': dict(path=[1], alpha=1., donor=0),
    'donor_self': dict(path=[0], alpha=.5, donor=0),
    'anchor_even': dict(path=[0, 1, 2, 3], alpha=.5, donor=None),
    'anchor_odd': dict(path=[0, 2, 1, 3], alpha=.5, donor=None),
    'direct_even': dict(path=[0, 1, 2, 3], alpha=1., donor=None),
    'direct_odd': dict(path=[0, 2, 1, 3], alpha=1., donor=None),
}


def validate_plan(plan, *, fixture_only=False):
    fixed = dict(schema=1, domain='dclm', status='registered_not_admitted', arms=['A', 'C'],
                 phase_updates=[768, 384, 768], batch=32, training_prefix_rows=61440,
                 lr=.0006, lr_ramp_updates=128, betas=[.9, .95], eps=1e-8,
                 weight_decay=0., clip_grad_norm=1.,
                 anchor_alpha=dict(donor_start=.05, donor_end=.5, stack=.5),
                 control_alpha=1., donor_detached=True,
                 train_paths=[list(p) for p in TRAIN_PATHS],
                 held_out_complete_paths=[list(p) for p in HELD_OUT_PATHS],
                 snapshots_global=[0, 768, 1152, 1344, 1536, 1920], primary_measurement_step=1920,
                 calibration_rows=256, measurement_rows=512, history_overlap_unknown=True,
                 remote_launch=False, weights_node_only=True,
                 initialization=dict(kind='scratch', linear_embedding_std=.02, rmsnorm_weight=1.,
                                     shared_embedding_readout=True, transformer_stem=False, transformer_coda=False))
    if any(plan.get(k) != v for k, v in fixed.items()):
        raise ValueError('Pilot computation/budget differs from registration')
    config = Config(**plan['model'])
    if not fixture_only and config != Config():
        raise ValueError('Production model dimensions differ')
    if fixture_only and (config.d > 64 or config.length > 64):
        raise ValueError('Fixture must remain small')
    if set(plan['seed_panels']) != {'0', '1'}:
        raise ValueError('Both registered seeds required')
    prior_pin = plan['data_provenance_source']
    if sha(prior_pin['path']) != prior_pin['sha256']:
        raise ValueError('Prior data provenance hash mismatch')
    previous = json.loads(Path(prior_pin['path']).read_text())['seed_panels']
    for seed, panel in plan['seed_panels'].items():
        if panel['seed'] != int(seed): raise ValueError('Seed identity mismatch')
        for k in ('data_recipe', 'expected_inputs', 'row_reservation'):
            if panel[k] != previous[seed][k]:
                raise ValueError('Registered data pins differ from prior provenance')
    return config


def references(model, x):
    """Full hidden/loss; gradient first32 values and norm per active parameter.

    Gradient summaries are partial checks, not full-gradient equality. Complete
    optimizer/fork equality must be checked separately during GPU qualification.
    """
    model.eval()
    result = {}
    for label, case in REFERENCE_CASES.items():
        model.zero_grad(set_to_none=True)
        with torch.no_grad(): h = model.hidden(x, **case)
        loss = model.nll(x, **case)
        if not torch.isfinite(loss).all(): raise ValueError('Nonfinite reference loss')
        loss.mean().backward()
        gradients, inactive = {}, []
        for name, p in model.named_parameters():
            if p.grad is None:
                inactive.append(name)
            else:
                if not torch.isfinite(p.grad).all(): raise ValueError('Nonfinite reference gradient')
                gradients[name] = dict(first32=p.grad.flatten()[:32].detach().cpu().clone(),
                                       norm=p.grad.detach().double().norm().cpu())
        result[label] = dict(hidden=h.detach().cpu(), nll=loss.detach().cpu(),
                             gradients=gradients, inactive=inactive)
    model.zero_grad(set_to_none=True)
    return result


def compare_references(first, second, policy):
    if set(first) != set(second): raise ValueError('Reference cell mismatch')
    maximum = {}
    for label, a in first.items():
        b = second[label]
        if a['inactive'] != b['inactive'] or a['gradients'].keys() != b['gradients'].keys():
            raise ValueError('Gradient participation mismatch')
        pairs = [(k, a[k], b[k]) for k in ('hidden', 'nll')]
        pairs += [(name + '/' + k, grad[k], b['gradients'][name][k])
                  for name, grad in a['gradients'].items() for k in ('first32', 'norm')]
        maximum[label] = 0.
        for name, x, y in pairs:
            torch.testing.assert_close(x, y, atol=policy['component_atol'], rtol=policy['component_rtol'],
                                       msg=lambda msg: label + '/' + name + ': ' + msg)
            maximum[label] = max(maximum[label], float((x - y).abs().max()))
    return maximum


def prepare(plan_path, plan_sha, out, *, fixture_only=False):
    if sha(plan_path) != plan_sha or Path(out).exists():
        raise ValueError('Pinned plan and fresh output directory required')
    plan = json.loads(Path(plan_path).read_text())
    config = validate_plan(plan, fixture_only=fixture_only)
    out = Path(out); out.mkdir(parents=True)
    receipts, previous_ids = {}, None
    for seed in (0, 1):
        panel = plan['seed_panels'][str(seed)]
        data = BoundRows(panel['data_recipe'], seed=seed, length=config.length, fixture_only=fixture_only)
        if data.binding['input_pins'] != panel['expected_inputs']:
            raise ValueError('Bound inputs differ from registered panel')
        prior = json.loads(Path(panel['data_recipe']['path']).read_text())['config']
        if not fixture_only and prior['row_reservation'] != panel['row_reservation']:
            raise ValueError('Reservation binding differs')
        if previous_ids is not None and np.intersect1d(previous_ids, data.train_ids).size:
            raise ValueError('Seed training rows overlap')
        previous_ids = data.train_ids.copy()
        t = CPUStreamTrainer(config, data, seed=seed)
        directory = out / ('s' + str(seed)); directory.mkdir()
        initial = directory / 'initial-fullstate.pt'
        initial_sha = t.save(initial)
        digest = fingerprint(t.model)
        x = data.batch(data.calibration_ids[:1])
        with measurement_runtime('cpu') as policy:
            first = references(t.model, x)
            repeat = references(t.model, x)
            maxima = compare_references(first, repeat, policy)
        if fingerprint(t.model) != digest or t.step != 0 or t.optimizer.state:
            raise ValueError('Preparation changed initial weights/optimizer')
        torch.save(first, directory / 'cpu-reference.pt')
        meta = dict(seed=seed, fixture_only=fixture_only, model=asdict(config), source=sources(),
                    data_binding=data.binding, plan_sha256=plan_sha, reference_cases=REFERENCE_CASES,
                    reference_row=int(data.calibration_ids[0]), initial_sha256=initial_sha,
                    initial_fingerprint=digest, reference_sha256=sha(directory / 'cpu-reference.pt'),
                    policy=policy, repeat_max_abs=maxima, train_updates=0,
                    gradient_scope='first32+norm per active parameter; not full gradient parity')
        (directory / 'config.json').write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
        receipts[str(seed)] = dict(config_sha256=sha(directory / 'config.json'),
                                  initial_sha256=initial_sha, reference_sha256=meta['reference_sha256'],
                                  reference_cells=len(first), train_updates=0)
        print('ANCHOR_CPU_SEED_COMPLETE', seed, json.dumps(receipts[str(seed)]), flush=True)
        del t, data, first, repeat
    result = dict(schema=1, plan_sha256=plan_sha, source=sources(), seeds=receipts,
                  fixture_only=fixture_only, cpu_references_complete=True,
                  train_updates=0, gpu_admitted=False)
    (out / 'prepared.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True); p.add_argument('--sha', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    print('ANCHOR_CPU_PREPARED', json.dumps(prepare(args.plan, args.sha, args.out)), flush=True)


if __name__ == '__main__': main()
