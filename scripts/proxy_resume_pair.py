"""Continue the original seed-1042 A/P6-iso pair, without a cooldown fork."""
import argparse
import copy
import math
from pathlib import Path

from run_experiments import load_jobs
from scripts.proxy_time_match import read, write
from scripts.stage_deep_kv_resume import digest

ARMS = ('A', 'P6-iso')
SOURCE_STEP, END_STEP, SEED = 10000, 20000, 1042


def make(source_run, output):
    source_run, output = Path(source_run).resolve(), Path(output).resolve()
    assert read(source_run/'run.json')['status'] == 'ok'
    original = read(source_run/'jobs.snapshot.json')['jobs']
    sources, training = {}, []
    for arm in ARMS:
        folder = source_run/f'seed-{SEED}'/arm
        cfg, result = read(folder/'train_config.json'), read(folder/'result.json')
        assert read(source_run/f'seed-{SEED}-validate-{arm}.json')['status'] == 'passed'
        assert cfg['pilot']['arm'] == result['arm'] == arm
        assert result['global_step'] == SOURCE_STEP and result['schedule_steps'] == 28600
        assert result['training_cost']['optimizer_steps'] == SOURCE_STEP
        assert 'cooldown' not in result
        assert not any(cfg['pilot'].get(k) is not None for k in ('cooldown_start', 'cooldown_steps'))
        assert not cfg['pilot'].get('proxy_sync_optimization', False)
        assert cfg['training']['seed'] == cfg['training']['data_seed'] == SEED
        assert cfg['training']['max_steps'] == 28600 and cfg['training']['warmup_steps'] == 1430
        assert cfg['training']['save_steps'] == 250 and cfg['training']['save_total_limit'] == 2
        assert cfg['world_size'] == 8 and not cfg['training']['ignore_data_skip']
        candidates = [j for j in original if j['name'] == f'seed-{SEED}-arm-{arm}']
        assert len(candidates) == 1
        job = copy.deepcopy(candidates[0]); argv = job['argv']
        assert job['gpus'] == list(range(8))
        assert not any(k in argv for k in ('--resume_from_checkpoint', '--cooldown_start', '--cooldown_steps'))
        assert argv[argv.index('--output_dir')+1] == f'{{run_dir}}/seed-{SEED}/{arm}'
        assert argv[argv.index('--stop_after')+1] == str(SOURCE_STEP)
        assert argv[argv.index('--save_total_limit')+1] == '2'
        argv[argv.index('--stop_after')+1] = str(END_STEP)
        argv[argv.index('--save_total_limit')+1] = '0'
        argv += ['--resume_from_checkpoint', f'{{run_dir}}/seed-{SEED}/{arm}/checkpoint-{SOURCE_STEP}']
        job['name'] = f'seed-{SEED}-resume-{arm}'
        job['required_outputs'] = [dict(path=f'seed-{SEED}/{arm}/result.json',
            json_equals=dict(arm=arm, global_step=END_STEP, status='stopped'))]
        sources[arm] = str(folder)
        training.append(job)
    output.mkdir(parents=True, exist_ok=True)
    sources_path = output/'sources.json'; write(sources_path, sources)
    items = [dict(name='stage-original-checkpoints',
        argv=['{python}', '-m', 'scripts.stage_deep_kv_resume', '--sources', str(sources_path),
              '--destination', f'{{run_dir}}/seed-{SEED}', '--step', str(SOURCE_STEP)],
        required_outputs=[dict(path=f'seed-{SEED}/resume_inputs.json', json_equals=dict(status='ok'))])]
    for arm, job in zip(ARMS, training):
        items.append(job)
        name = f'validate-{arm}'
        items.append(dict(name=name, argv=['{python}', '-m', 'scripts.check_fa4_proxy', 'validate',
            '--run-dir', '{run_dir}', '--steps', str(END_STEP), '--seed', str(SEED), '--arms', arm,
            '--attention-backend', 'fa4', '--output', '{run_dir}/'+name+'.json'],
            required_outputs=[dict(path=name+'.json', json_equals=dict(status='passed'))]))
    items.append(dict(name='validate-continuations', argv=['{python}', '-m', 'scripts.proxy_resume_pair',
        'report', '--source-run', str(source_run), '--output', '{run_dir}'],
        required_outputs=[dict(path='continuation-summary.json', json_equals=dict(status='passed'))]))
    write(output/'jobs.json', dict(jobs=items)); load_jobs(output/'jobs.json')
    return dict(source_step=SOURCE_STEP, stop_after=END_STEP, seed=SEED, arms=list(ARMS),
                schedule_steps=28600, warmup_steps=1430, save_steps=250, save_total_limit=0,
                cooldown=False, stages=len(items))


def report(source_run, output):
    from deep_kv.report import report as compare
    from train import resume_performance_changes
    source_run, output = Path(source_run), Path(output)
    root = output/f'seed-{SEED}'
    receipt = read(root/'resume_inputs.json')
    assert receipt['status'] == 'ok' and receipt['source_step'] == SOURCE_STEP
    assert set(receipt['arms']) == set(ARMS)
    compare(root, arms=ARMS, expected_step=END_STEP, expected_seed=SEED,
            expected_attention_backend='fa4')
    rows = []
    for arm in ARMS:
        folder, source = root/arm, source_run/f'seed-{SEED}'/arm
        assert read(output/f'validate-{arm}.json')['status'] == 'passed'
        result, previous = read(folder/'result.json'), read(source/'result.json')
        cfg, old = read(folder/'train_config.json'), read(source/'train_config.json')
        expected_change = {'save_total_limit': dict(before=2, after=0)}
        assert resume_performance_changes(old, cfg) == expected_change
        transitions = list(folder.glob('resume-transition-*.json'))
        assert len(transitions) == 1 and read(transitions[0])['changes'] == expected_change
        assert 'cooldown' not in result and result['global_step'] == END_STEP
        assert result['training_cost']['optimizer_steps'] == END_STEP-SOURCE_STEP
        checkpoints = {int(p.name.split('-')[-1]) for p in folder.glob('checkpoint-*')}
        assert checkpoints == set(range(SOURCE_STEP, END_STEP+1, 250))
        # Originals and their staged copies must both remain byte-identical.
        assert receipt['arms'][arm]['source'] == str(source)
        for name, expected in receipt['arms'][arm]['files'].items():
            for base in (source, folder):
                path = base/f'checkpoint-{SOURCE_STEP}'/name
                assert path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256']
        runtime = previous['training_cost']['runtime_seconds']+result['training_cost']['runtime_seconds']
        loss = result['evaluation']['eval_lm_loss']
        assert math.isfinite(runtime) and runtime > 0 and math.isfinite(loss)
        rows.append(dict(arm=arm, step=END_STEP, eval_lm_loss=loss, perplexity=math.exp(loss),
                         cumulative_runtime_seconds=runtime, retained_checkpoints=sorted(checkpoints)))
    value = dict(status='passed', source_run=str(source_run), source_step=SOURCE_STEP,
                 stop_after=END_STEP, seed=SEED, cooldown=False, results=rows)
    write(output/'continuation-summary.json', value)
    return value


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('make', 'report'))
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(globals()[args.mode](args.source_run, args.output), indent=2))
