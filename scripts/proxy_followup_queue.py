"""Nine authorized 2,500-step runs, using the existing Trainer and job runner."""
import json
from pathlib import Path

from deep_kv.__main__ import jobs
from run_experiments import load_jobs

GROUPS = (
    (42, ('P6-iso-sparse', 'P6-iso-short', 'P6-iso-weighted',
          'P6-iso-layernorm', 'P7-simple-sparse', 'P7-simple-short')),
    (1042, ('A', 'P6-iso', 'P7-simple')),
)


def make(recipe_path, output):
    items = []
    for seed, arms in GROUPS:
        generated = jobs(recipe_path, stop_after=2500, arms=arms, seeds=[seed])['jobs']
        for item in generated:
            # Each group already has a per-seed comparison. A cross-seed report
            # would be misleading here because the two groups contain different arms.
            if item['name'] == 'compare-seeds':
                continue
            if 'gpus' in item:
                argv = item['argv']
                arm = argv[argv.index('--arm') + 1]
                if '--proxy_module_seed' in argv:
                    raise ValueError('Recipe must leave module seed to this per-run queue')
                if arm != 'A':
                    argv += ['--proxy_module_seed', str(seed + 1)]
                # Set this before Python starts; Torch/Trainer/data seeds are
                # already explicit in the generated argv. A has no proxy branch.
                item['argv'] = ['env', f'PYTHONHASHSEED={seed}', *argv]
                items.append(item)
                name = f'seed-{seed}-validate-{arm}'
                items.append(dict(name=name, argv=[
                    '{python}', '-m', 'scripts.check_fa4_proxy', 'validate',
                    '--run-dir', '{run_dir}', '--steps', '2500', '--seed', str(seed),
                    '--arms', arm, '--attention-backend', 'fa4',
                    '--output', '{run_dir}/' + name + '.json'],
                    required_outputs=[dict(path=name+'.json', json_equals={'status':'passed'})]))
            else:
                items.append(item)
    output = Path(output)
    with output.open('x') as handle:
        json.dump(dict(jobs=items), handle, indent=2)
    load_jobs(output)
    return items
