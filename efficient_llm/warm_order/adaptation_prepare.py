"""Explicit CPU preparation from actual parents and corpus; never a launcher."""
import json
import math
from pathlib import Path

import torch

from .adaptation_train import load_parent, source_hashes, POLICY
from .adaptation_reservation import reserve, bind, validate_binding, verify
from .data import sha
from .horizon_train import write_json


def prepare(plan_path, plan_sha256, parent_registration_path, parent_directories,
            snapshots, threads, out):
    out = Path(out)
    if out.exists(): raise ValueError('Fresh preparation output required')
    if sha(plan_path) != plan_sha256: raise ValueError('Preparation plan hash mismatch')
    plan = json.loads(Path(plan_path).read_text())
    if sha(parent_registration_path) != plan['parent_registration_sha256']:
        raise ValueError('Parent registration hash mismatch')
    registration = json.loads(Path(parent_registration_path).read_text())
    if set(parent_directories) != {'0','1'}: raise ValueError('Both explicit parent directories required')
    if type(threads) is not int or threads <= 0: raise ValueError('Positive CPU thread count required')
    if (not snapshots or any(type(s) is not int or s < 0 for s in snapshots) or
            snapshots != sorted(set(snapshots)) or snapshots[0] != 0 or snapshots[-1] != plan['steps']):
        raise ValueError('Registered snapshot schedule must span zero to terminal')
    receipts = {}; config_common = None
    # Preserve caller RNG. Load and release one parent/fullstate/optimizer at a time.
    with torch.random.fork_rng(devices=[]):
        for seed in (0,1):
            registered = registration['parents'][f'D-s{seed}']; pins = registered['file_pins']
            model, optimizer, state, names = load_parent(parent_directories[str(seed)], pins)
            c = state['recipe']['config']
            if (c['seed'] != seed or c['batch'] != plan['batch'] or model.c.length != plan['length'] or
                    c['model'] != registered['model']):
                raise ValueError('Actual parent seed/batch/model mismatch')
            lr = optimizer.param_groups[0]['lr']
            if not math.isclose(lr, registered['terminal_lr'], abs_tol=1e-12, rel_tol=0):
                raise ValueError('Registered terminal LR mismatch')
            for suffix in ('.bin','.starts'):
                if c['expected_inputs'][c['data']+suffix] != plan['expected_stream_hashes'][plan['data']+suffix]:
                    raise ValueError('Parent and continuation stream identity mismatch')
            if config_common is None: config_common = c['model']
            elif config_common != c['model']: raise ValueError('Parent model configurations differ')
            receipts[str(seed)] = dict(parent=str(parent_directories[str(seed)]), file_pins=pins,
                parent_step=state['step'], terminal_lr=lr, parameter_names=names, actual_fullstate_loaded=True)
            del model, optimizer, state
    # reserve requires actual stream bytes and preserves partial output on later failures.
    reserve(plan, out/'reservation')
    write_json(out/'plan.json', plan)
    write_json(out/'parent_checks.json', receipts)
    files = {name:sha(out/name) for name in ('plan.json','parent_checks.json','reservation/manifest.json')}
    reservation_sha = files['reservation/manifest.json']
    for seed in (0,1):
        pair = []
        for branch in ('A','B'):
            c = dict(branch=branch, parent=str(parent_directories[str(seed)]),
                     parent_hashes=receipts[str(seed)]['file_pins'], steps=plan['steps'], batch=plan['batch'],
                     threads=threads, domain='dclm', lr_policy=POLICY, snapshots=snapshots, data=plan['data'])
            c = bind(c, out/'reservation', reservation_sha, seed); validate_binding(c)
            name = f'{branch}-s{seed}.json'; write_json(out/name,c); files[name] = sha(out/name)
            pair.append({k:v for k,v in c.items() if k != 'branch'})
        if pair[0] != pair[1]: raise ValueError('Unmatched A/B preparation')
    report = dict(schema=1,status='CPU_PREPARED_NOT_GPU_ADMITTED',files=files,training_source=source_hashes(),
                  preparation_source_sha256=sha(__file__),model=config_common,source_plan_sha256=plan_sha256,
                  parent_registration_sha256=plan['parent_registration_sha256'],
                  actual_parents_loaded=2,train_jobs_launched=0,
                  current_gpu_ownership_verified=False,gpu_numerical_gates_verified=False,
                  production_ready=False,weights_exported=False,
                  flags=['exploratory','historical_document_overlap_unknown','CPU_preparation_only'])
    write_json(out/'manifest.json',report)
    verify(out/'reservation')
    return report
