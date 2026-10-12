"""Continue a stopped expanded study with seed 42 and a fixed 1e-5 rate.

Reuses only successful, hash-verified source artifacts. Interrupted fits restart
in the fresh destination; original runs and their metadata remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

from run_experiments import load_jobs, verify_outputs
from scripts.expanded_evaluation_study import make
from scripts.finetune_study import write


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def flag(argv, name):
    return argv[argv.index(name) + 1]


def verified_job(job, record, root, evidence):
    if record.get('status') != 'ok' or record.get('returncode') != 0:
        raise ValueError('Only completed jobs can be reused: ' + job['name'])
    actual = verify_outputs(job, root)
    if actual != record['artifacts']:
        raise ValueError('Source artifacts changed: ' + job['name'])
    evidence.extend(dict(path=str(root / a['path']), sha256=a['sha256']) for a in actual)


def verify_reuse(receipt, output):
    report = read(receipt)
    for artifact in report['artifacts']:
        if digest(artifact['path']) != artifact['sha256']:
            raise ValueError('Reused artifact changed: ' + artifact['path'])
    write(output, dict(status='passed', artifacts=len(report['artifacts']),
                       receipt_sha256=digest(receipt)))


def build(source, destination, project):
    source, destination, project = map(lambda p: Path(p).resolve(), (source, destination, project))
    stop = read(source / 'reduce-seed42-stop.json')
    terminal = read(source / 'supervised/supervisor.json')
    if (not terminal.get('finished_at') or 'handoff_error' in terminal
            or not stop['burn_handoff'].get('collective_progress_verified')
            or stop['previous_supervisor_final'] != terminal):
        raise ValueError('Require completed owned stop and verified burn handoff')
    if destination.exists():
        raise FileExistsError(destination)
    old_run = source / 'supervised/run'
    old_fit = old_run / 'finetune'
    old_jobs = {j['name']: j for j in load_jobs(old_run / 'jobs.snapshot.json')}
    old_records = {j['name']: j for j in read(old_run / 'run.json')['jobs']}
    fit_jobs = {j['name']: j for j in load_jobs(old_fit / 'jobs.snapshot.json')}
    fit_records = {j['name']: j for j in read(old_fit / 'run.json')['jobs']}
    evidence = []
    completed_outer = ('numerics-A20', 'numerics-P20', 'expanded-smoke', 'expanded-full')
    for name in completed_outer:
        verified_job(old_jobs[name], old_records[name], old_run, evidence)
    args = old_jobs['paired-plan']['argv']
    checkpoints = read(source / 'study-scope.json')['checkpoints']
    legacy = flag(args, '--legacy-f') if '--legacy-f' in args else None
    make(destination, project, checkpoints, flag(args, '--dataset-root'), flag(args, '--manifest'),
         flag(args, '--protocol'), flag(args, '--position-sample'), legacy, seeds=(42,), rates=(1e-5,))
    fresh_run = destination / 'supervised/run'
    fresh_fit = fresh_run / 'finetune'
    plan_file = destination / 'finetune-plan/jobs.json'
    remaining, reused, fitted = [], [], {}
    for job in load_jobs(plan_file):
        name = job['name']
        record = fit_records.get(name, {})
        if name.startswith(('gate-', 'smoke-', 'verify-reload-', 'search-')) and record.get('status') == 'ok':
            previous = fit_jobs[name]
            verified_job(previous, record, old_fit, evidence)
            if name.startswith('search-'):
                old_config, new_config = read(previous['argv'][-1]), read(job['argv'][-1])
                if {k:v for k,v in old_config.items() if k != 'output_dir'} != {k:v for k,v in new_config.items() if k != 'output_dir'}:
                    raise ValueError('Training configuration differs: ' + name)
                result_path = old_fit / previous['required_outputs'][0]['path']
                result = read(result_path)
                weights = result_path.parent / 'best/model.safetensors'
                if (result['status'] != 'completed' or result['seed'] != 42
                        or result['learning_rate'] != 1e-5 or result['smoke'] or result['evaluate_run']
                        or digest(weights) != result['weight_sha256']):
                    raise ValueError('Invalid reusable fit: ' + name)
                evidence.append(dict(path=str(weights), sha256=result['weight_sha256']))
                evidence.append(dict(path=previous['argv'][-1], sha256=digest(previous['argv'][-1])))
                fitted[str(fresh_fit / job['required_outputs'][0]['path']).removesuffix('/result.json')] = str(result_path.parent)
            reused.append(name)
        else:
            remaining.append(job)
    # Fixed-rate selectors point at original complete fits when available.
    # All selectors are recomputed, never reuse a previous two-rate choice.
    for job in remaining:
        if job['name'].startswith('select-'):
            job['argv'] = [fitted.get(arg, arg) for arg in job['argv']]
    evidence.extend(dict(path=str(p), sha256=digest(p)) for p in (
        source/'reduce-seed42-stop.json', source/'supervised/supervisor.json',
        old_run/'run.json', old_fit/'run.json', old_fit/'jobs.snapshot.json'))
    receipt = destination / 'reuse.json'
    write(receipt, dict(status='verified', source=str(source), seeds=[42], rates=[1e-5],
                        completed_outer=list(completed_outer), reused=reused, artifacts=evidence,
                        remaining_fits=sum(j['name'].startswith('search-') for j in remaining)))
    plan_file.write_text(json.dumps(dict(jobs=remaining), indent=2))
    load_jobs(plan_file)
    outer = [j for j in load_jobs(destination/'jobs.json') if j['name'] not in completed_outer]
    outer.insert(0, dict(name='verify-reused-artifacts', argv=['{python}', '-m', 'scripts.reduce_expanded_study',
        'verify', '--receipt', str(receipt), '--output', '{run_dir}/reuse-verified.json'], timeout_seconds=1800,
        required_outputs=[dict(path='reuse-verified.json', json_equals=dict(status='passed'))]))
    (destination/'jobs.json').write_text(json.dumps(dict(jobs=outer), indent=2))
    load_jobs(destination/'jobs.json')
    scope = read(destination/'study-scope.json')
    scope.update(order=[j['name'] for j in outer], fit_stages=len(remaining),
                 reused_source=str(source), remaining_fits=sum(j['name'].startswith('search-') for j in remaining))
    (destination/'study-scope.json').write_text(json.dumps(scope, indent=2))
    return read(receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    b=sub.add_parser('build')
    for key in ('source','destination','project'):b.add_argument('--'+key,required=True)
    v=sub.add_parser('verify');v.add_argument('--receipt',required=True);v.add_argument('--output',required=True)
    a=p.parse_args()
    if a.mode=='build':
        result=build(a.source,a.destination,a.project)
        print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}),flush=True)
    else:verify_reuse(a.receipt,a.output)


if __name__=='__main__':main()
