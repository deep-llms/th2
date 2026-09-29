"""Observe a native Trainer checkpoint resume, then compare old/optimized updates.

This is a smoke-test wrapper only: production jobs still call train.py directly.
It never changes Trainer loading, data skipping, optimization or RNG handling.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

FAST = dict(checkpoint_layers=False, checkpoint_lm=False, checkpoint_aux=False,
            causal_attention=False, lm_chunk=128)


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def state_hash(value):
    import numpy as np
    import torch
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((tuple(item.shape), item.dtype)).encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.shape, item.dtype)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for entry in item:
                visit(entry)
        else:
            digest.update(repr(item).encode())
        digest.update(b'\0')
    visit(value)
    return digest.hexdigest()


def worker(receipt_prefix, arguments):
    import random
    import numpy as np
    import torch
    from safetensors.torch import load_file
    import train

    evidence = dict(batches=[])

    class CheckedTrainer(train.DeepKVTrainer):
        def _load_optimizer_and_scheduler(self, checkpoint):
            super()._load_optimizer_and_scheduler(checkpoint)
            source = Path(checkpoint)
            for name, actual in [('optimizer', self.optimizer.state_dict()),
                                 ('scheduler', self.lr_scheduler.state_dict())]:
                expected = torch.load(source / f'{name}.pt', map_location='cpu', weights_only=True)
                evidence[name] = state_hash(actual)
                assert evidence[name] == state_hash(expected), f'{name} restoration differs'
            evidence['checkpoint'] = str(source)
            evidence['source_step'] = json.loads((source / 'trainer_state.json').read_text())['global_step']
            evidence['source_model'] = state_hash(load_file(str(source / 'model.safetensors')))

        def _load_rng_state(self, checkpoint):
            super()._load_rng_state(checkpoint)
            # These are our own trusted training checkpoints, including NumPy state.
            suffix = f'_{self.args.process_index}' if self.args.world_size > 1 else ''
            expected = torch.load(Path(checkpoint) / f'rng_state{suffix}.pth', weights_only=False)
            actual = dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state())
            if 'cuda' in expected:
                actual['cuda'] = (torch.cuda.get_rng_state_all() if self.args.world_size > 1
                                  else torch.cuda.get_rng_state())
            assert state_hash(actual) == state_hash(expected), 'RNG restoration differs'
            evidence['rng'] = state_hash(actual)

        def compute_loss(self, model, inputs, *args, **kwargs):
            if model.training:
                assert self.state.global_step == evidence['source_step']
                assert 'rng' in evidence, 'Training started before RNG restoration'
                if not evidence['batches']:
                    restored = state_hash(self.accelerator.unwrap_model(model).state_dict())
                    assert restored == evidence['source_model'], 'Model restoration differs'
                    evidence['model'] = restored
                evidence['batches'].append(state_hash(inputs))
            return super().compute_loss(model, inputs, *args, **kwargs)

        def train(self, *args, **kwargs):
            result = super().train(*args, **kwargs)
            evidence.update(global_step=self.state.global_step, world_size=self.args.world_size,
                            rank=self.args.process_index, ignore_data_skip=self.args.ignore_data_skip,
                            accumulation=self.args.gradient_accumulation_steps,
                            scheduler_final=state_hash(self.lr_scheduler.state_dict()))
            assert evidence['global_step'] == evidence['source_step'] + 1
            assert len(evidence['batches']) == evidence['accumulation']
            assert not evidence['ignore_data_skip']
            return result

    train.DeepKVTrainer = CheckedTrainer
    sys.argv = ['train.py', *arguments]
    train.main()
    evidence['status'] = 'ok'
    write(f'{receipt_prefix}-{evidence["rank"]}.json', evidence)


def relative_difference(left, right):
    import torch
    delta = norm = 0.
    assert left.keys() == right.keys()
    for key in left:
        a, b = left[key].double(), right[key].double()
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        delta += (a - b).square().sum().item()
        norm += a.square().sum().item()
    return (delta / max(norm, 1e-100)) ** .5


def compare(root, step, arms='BFG', world=8, control_root=None):
    import torch
    from safetensors.torch import load_file
    results = {}
    for arm in arms:
        control = (control_root or root / 'control') / arm
        fast = root / 'optimized' / arm
        for rank in range(world):
            a, b = [json.loads((p / f'resume-check-{rank}.json').read_text()) for p in (control, fast)]
            for record in (a, b):
                assert record['status'] == 'ok' and record['world_size'] == world
                assert record['rank'] == rank and record['global_step'] == step + 1
                assert record['source_step'] == step and not record['ignore_data_skip']
                assert len(record['batches']) == record['accumulation']
            for key in ('batches', 'optimizer', 'scheduler', 'model', 'rng', 'scheduler_final'):
                assert a[key] == b[key], f'{arm} rank {rank}: {key} differs'
        configs = [json.loads((p / 'train_config.json').read_text()) for p in (control, fast)]
        from train import resume_performance_changes
        changes = resume_performance_changes(*configs, allow=True)
        assert set(changes) == {k for k, v in FAST.items() if configs[0]['pilot'][k] != v}
        assert all(configs[1]['pilot'][k] == v for k, v in FAST.items())
        assert len(list(fast.glob('resume-transition-*.json'))) == 1
        ends = [p / f'checkpoint-{step + 1}' for p in (control, fast)]
        weights = [load_file(str(p / 'model.safetensors')) for p in ends]
        relative = relative_difference(*weights)
        original = load_file(str(control / f'checkpoint-{step}/model.safetensors'))
        update = relative_difference(original, weights[0])
        assert relative <= 1e-5 and relative <= .05 * max(update, 1e-15), (arm, relative, update)
        del weights, original
        optimizers = [torch.load(p / 'optimizer.pt', map_location='cpu', weights_only=True) for p in ends]
        assert optimizers[0]['param_groups'] == optimizers[1]['param_groups']
        moment_errors = {}
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            states = [{idx: state[key] for idx, state in opt['state'].items()} for opt in optimizers]
            if key == 'step':
                assert state_hash(states[0]) == state_hash(states[1])
                assert all(int(value) == step + 1 for value in states[0].values())
            else:
                moment_errors[key] = relative_difference(*states)
                assert moment_errors[key] <= .03, (arm, key, moment_errors[key])
        del optimizers, states
        losses = []
        for path, cfg, checkpoint in zip((control, fast), configs, ends):
            result = json.loads((path / 'result.json').read_text())
            assert result['global_step'] == step + 1
            assert result['schedule_steps'] == cfg['training']['max_steps']
            assert result['input_tokens'] == (step + 1) * cfg['tokens_per_update']
            assert result['evaluation']['eval_rows'] == cfg['data']['eval_rows']
            assert len(list(checkpoint.glob('rng_state*.pth'))) == world
            losses.append(result['evaluation']['eval_lm_loss'])
        assert abs(losses[0] - losses[1]) <= .001, (arm, losses)
        results[arm] = dict(parameter_relative_l2=relative, control_update_relative_l2=update,
                            moment_relative_l2=moment_errors, eval_lm_losses=losses)
    write(root / 'verified.json', dict(status='ok', source_step=step, arms=results,
                                      control_root=str((control_root or root / 'control').resolve())))
    print('REAL_CHECKPOINT_RESUME_VERIFIED', json.dumps(results), flush=True)


def make_jobs(recipe_path, root, source_root, step, end, control_root=None):
    from deep_kv.__main__ import jobs
    root.mkdir(parents=True, exist_ok=True)
    sources = root / 'resume-sources.json'
    write(sources, {arm: str(source_root / arm) for arm in 'BFG'})
    recipe = json.loads(recipe_path.read_text())
    optimized = root / 'optimized-recipe.json'
    write(optimized, {**recipe, **FAST, 'allow_performance_change_on_resume': True})
    items = []
    def staging(destination, name):
        return dict(name=name, argv=['{python}', '-m', 'scripts.stage_deep_kv_resume',
            '--sources', str(sources), '--destination', destination, '--step', str(step)],
            required_outputs=[dict(path=destination.replace('{run_dir}/', '') + '/resume_inputs.json',
                                   json_equals=dict(status='ok', source_step=step))])
    for mode, config in [('control', recipe_path), ('optimized', optimized)]:
        if mode == 'control' and control_root is not None:
            continue  # Reuse completed, unchanged controls; compare still checks all their state/data receipts.
        prefix = '{run_dir}/resume-smoke/' + mode
        items.append(staging(prefix, f'stage-smoke-{mode}'))
        for job in jobs(config, step + 1, 'BFG')['jobs'][:-1]:
            arm = job['name'][-1]
            job['name'] = f'smoke-{mode}-{arm}'
            job['argv'] = [arg.replace('{run_dir}', prefix) for arg in job['argv']]
            index = next(i for i, arg in enumerate(job['argv']) if arg.endswith('/train.py'))
            job['argv'][index:index + 1] = ['-m', 'scripts.check_optimized_resume', 'worker',
                                           '--receipt-prefix', prefix + f'/{arm}/resume-check']
            job['argv'] += ['--resume_from_checkpoint', prefix + f'/{arm}/checkpoint-{step}']
            for output in job['required_outputs']:
                output['path'] = f'resume-smoke/{mode}/' + output['path']
            items.append(job)
    items.append(dict(name='verify-resume', argv=['{python}', '-m', 'scripts.check_optimized_resume',
        'compare', '--root', '{run_dir}/resume-smoke', '--step', str(step)],
        required_outputs=[dict(path='resume-smoke/verified.json', json_equals=dict(status='ok'))]))
    if control_root is not None:
        items[-1]['argv'] += ['--control-root', str(control_root)]
    items.append(staging('{run_dir}/continuation', 'stage-continuation'))
    for job in jobs(optimized, end, 'BFG')['jobs']:
        job['argv'] = [arg.replace('{run_dir}', '{run_dir}/continuation') for arg in job['argv']]
        for output in job['required_outputs']:
            output['path'] = 'continuation/' + output['path']
        if job['name'].startswith('arm-'):
            job['argv'] += ['--resume_from_checkpoint', '{run_dir}/continuation/' + job['name'][-1] + f'/checkpoint-{step}']
        items.append(job)
    write(root / 'jobs.json', dict(jobs=items))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('worker')
    p.add_argument('--receipt-prefix', required=True)
    p = sub.add_parser('compare')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--control-root', type=Path)
    p = sub.add_parser('make-jobs')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--end', type=int, required=True)
    p.add_argument('--control-root', type=Path)
    args, extra = parser.parse_known_args()
    if args.command == 'worker':
        worker(args.receipt_prefix, extra)
    else:
        if extra:
            parser.error(f'Unknown arguments: {extra}')
        if args.command == 'compare':
            compare(args.root, args.step, control_root=args.control_root)
        else:
            make_jobs(args.recipe, args.root, args.source_root, args.step, args.end, args.control_root)


if __name__ == '__main__':
    main()
