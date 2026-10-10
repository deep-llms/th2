"""Resume a verified A checkpoint to approximately match P6-iso Trainer runtime."""
import argparse
import copy
import json
import math
from pathlib import Path

from run_experiments import load_jobs


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def timing_plan(a, reference):
    assert a['arm'] == 'A' and reference['arm'] == 'P6-iso'
    step = a['global_step']
    assert step == reference['global_step'] and step > 0
    assert a['schedule_steps'] == reference['schedule_steps']
    for item in (a, reference):
        assert item['training_cost']['optimizer_steps'] == step
        assert math.isfinite(item['training_cost']['runtime_seconds'])
        assert item['training_cost']['runtime_seconds'] > 0
    base = a['training_cost']['runtime_seconds']
    budget = reference['training_cost']['runtime_seconds']
    end = round(step * budget / base)
    assert step < end < a['schedule_steps']
    return dict(source_step=step, stop_after=end, baseline_runtime_seconds=base,
                reference_runtime_seconds=budget, additional_budget_seconds=budget-base,
                basis='Trainer train_runtime; excludes preprocessing and final evaluation',
                matching='Estimated step cutoff; report actual cumulative runtime after resume')


def make(source_run, output):
    source_run, output = Path(source_run).resolve(), Path(output).resolve()
    assert read(source_run/'run.json')['status'] == 'ok'
    a_dir, ref_dir = (source_run/'seed-1042'/arm for arm in ('A','P6-iso'))
    a, reference = read(a_dir/'result.json'), read(ref_dir/'result.json')
    plan = timing_plan(a, reference)
    step, end = plan['source_step'], plan['stop_after']
    assert step == 10000 and a['schedule_steps'] == 28600
    for arm in ('A', 'P6-iso'):
        assert read(source_run/f'seed-1042-validate-{arm}.json')['status'] == 'passed'
    cfg = read(a_dir/'train_config.json')
    assert cfg['training']['seed'] == cfg['training']['data_seed'] == 1042
    assert cfg['training']['save_steps'] == 250 and cfg['world_size'] == 8
    assert not cfg['training']['ignore_data_skip']
    # Reuse the exact executed argv, changing only cutoff/retention/resume path.
    candidates = [j for j in read(source_run/'jobs.snapshot.json')['jobs']
                  if j['name'] == 'seed-1042-arm-A']
    assert len(candidates) == 1
    job = copy.deepcopy(candidates[0]); argv = job['argv']
    assert job['gpus'] == list(range(8)) and '--resume_from_checkpoint' not in argv
    assert argv[argv.index('--output_dir')+1] == '{run_dir}/seed-1042/A'
    assert argv[argv.index('--stop_after')+1] == str(step)
    argv[argv.index('--stop_after')+1] = str(end)
    argv[argv.index('--save_total_limit')+1] = '0'
    argv += ['--resume_from_checkpoint', f'{{run_dir}}/seed-1042/A/checkpoint-{step}']
    job['name'] = 'seed-1042-resume-A'
    job['required_outputs'] = [dict(path='seed-1042/A/result.json',
        json_equals=dict(arm='A', global_step=end, status='stopped'))]
    output.mkdir(parents=True, exist_ok=True)
    sources = output/'sources.json'; write(sources, {'A': str(a_dir)})
    write(output/'timing-plan.json', plan)
    items = [dict(name='stage-A-checkpoint', argv=['{python}','-m','scripts.stage_deep_kv_resume',
        '--sources',str(sources),'--destination','{run_dir}/seed-1042','--step',str(step)],
        required_outputs=[dict(path='seed-1042/resume_inputs.json',json_equals=dict(status='ok'))]), job,
        dict(name='validate-A', argv=['{python}','-m','scripts.check_fa4_proxy','validate',
            '--run-dir','{run_dir}','--steps',str(end),'--seed','1042','--arms','A',
            '--attention-backend','fa4','--output','{run_dir}/validate-A.json'],
            required_outputs=[dict(path='validate-A.json',json_equals=dict(status='passed'))]),
        dict(name='compare-time', argv=['{python}','-m','scripts.proxy_time_match','report',
            '--source-run',str(source_run),'--output','{run_dir}'],
            required_outputs=[dict(path='time-comparison.json',json_equals=dict(status='passed'))])]
    write(output/'jobs.json',dict(jobs=items)); load_jobs(output/'jobs.json')
    return plan


def report(source_run, output):
    source_run, output = Path(source_run), Path(output)
    a = read(source_run/'seed-1042/A/result.json')
    reference = read(source_run/'seed-1042/P6-iso/result.json')
    plan = timing_plan(a, reference)
    current = output/'seed-1042/A'
    result = read(current/'result.json'); cfg = read(current/'train_config.json')
    assert result['global_step'] == plan['stop_after']
    assert result['training_cost']['optimizer_steps'] == plan['stop_after']-plan['source_step']
    assert cfg['training']['save_total_limit'] == 0 and cfg['training']['save_steps'] == 250
    expected = {plan['source_step'],plan['stop_after'],
                *range((plan['source_step']//250+1)*250,plan['stop_after']+1,250)}
    observed = {int(p.name.split('-')[-1]) for p in current.glob('checkpoint-*')}
    assert observed == expected, (observed,expected)
    transitions = list(current.glob('resume-transition-*.json'))
    assert len(transitions) == 1
    assert read(transitions[0])['changes'] == {'save_total_limit':dict(before=2,after=0)}
    previous = read(source_run/'seed-1042/A/train_config.json')
    for key in ('train_fingerprint','eval_fingerprint','tokens_per_update','world_size'):
        assert previous[key] == cfg[key]
    actual = plan['baseline_runtime_seconds'] + result['training_cost']['runtime_seconds']
    relative_gap = actual/plan['reference_runtime_seconds']-1
    rows = []
    for item, runtime in ((a,plan['baseline_runtime_seconds']), (reference,plan['reference_runtime_seconds']),
                          (result,actual)):
        loss=item['evaluation']['eval_lm_loss']
        rows.append(dict(arm=item['arm'],step=item['global_step'],runtime_seconds=runtime,
                         eval_lm_loss=loss,perplexity=math.exp(loss)))
    value=dict(status='passed',plan=plan,actual_time_gap_fraction=relative_gap,
               within_one_percent=abs(relative_gap)<=.01,results=rows,
               retained_checkpoints=sorted(observed))
    write(output/'time-comparison.json',value)
    print(json.dumps(value,indent=2))
    return value


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('make','report'))
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='make': print(json.dumps(make(args.source_run,args.output),indent=2))
    else: report(args.source_run,args.output)
