"""Prepare Stage 0a, then gate D/P1-A/P1-B behind T12/T14 and the GPU supervisor."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)


def gated_jobs(prepared, accelerate_config):
    """The sequential runner halts at any failed check; its supervisor restores burns."""
    def job(name, argv, artifact, timeout):
        return dict(name=name, gpus=list(range(8)), argv=list(map(str, argv)),
                    timeout_seconds=timeout,
                    required_outputs=[dict(path=artifact, json_equals=dict(status='passed'))])
    jobs = [job('T12', ['{python}', ROOT/'tests/bdot/resume_gpu_probe.py', 'suite',
        '--output', '{run_dir}/T12', '--accelerate-config', accelerate_config,
        '--schedule-steps', '1000', '--steps', '200', '--split', '100', '--soft-split', '20',
        '--curriculum', 'B', '--expert-backend', 'grouped'], 'T12/summary.json', 2400),
        job('T14', ['{python}', ROOT/'scripts/bdot_pipeline_check.py',
            '--arguments', prepared/'D.train.json', '--output', '{run_dir}/T14',
            '--accelerate-config', accelerate_config], 'T14/summary.json', 3600)]
    pilots = json.loads((prepared/'jobs.json').read_text())['jobs']
    assert [j['name'] for j in pilots] == ['D', 'P1-A', 'P1-B']
    for j in pilots:
        j['timeout_seconds'] = 10800
    return dict(jobs=jobs+pilots)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--predecessor', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--session', required=True)
    a = p.parse_args()
    import socket
    if socket.gethostname() != a.host:
        raise ValueError('Wrong host')
    a.output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1',
        'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'WANDB_MODE': 'offline',
        'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    def run(script, *args, environment=env):
        print('PILOT_PREPARATION '+script, flush=True)
        subprocess.run([sys.executable, str(ROOT/script), *map(str, args)],
                       cwd=ROOT, env=environment, check=True)
    try:
        from scripts.native_baseline_pipeline import burn_gate
        from scripts.gpu_process import inspect
        from scripts.train_then_burn import GUARD_HASH, BURN_HASH, digest
        burn_gate(a.predecessor)
        assert json.loads((a.predecessor/'supervised/run/checks/summary.json').read_text())['status'] == 'passed'
        # ~250 GiB of pilot checkpoints plus the check pair, with ample headroom.
        if shutil.disk_usage(a.output).free < 500*1024**3:
            raise ValueError('Less than 500 GiB free for retained pilot checkpoints')
        run('scripts/verify_cx_english.py', '--data-root', a.data_root,
            '--model-root', a.model_root, '--output', a.output/'assets-verified.json')
        assets = json.loads((a.output/'assets-verified.json').read_text())
        prepared = a.output/'prepared'
        accelerate = ROOT/'resources/accelerate_b200.yaml'
        run('scripts/make_bdot_pilot.py', '--tokenizer', assets['tokenizer'],
            '--data', Path(assets['subset_root'])/'train',
            '--eval-data', Path(assets['subset_root'])/'validation',
            '--revision', assets['contract']['revision'], '--accelerate-config', accelerate,
            '--expert-backend', 'grouped', '--output', prepared)
        run('scripts/warm_training_cache.py', '--arguments', prepared/'D.train.json',
            '--output', a.output/'cache-ready.json')
        jobs = a.output/'gated-jobs.json'
        write(jobs, gated_jobs(prepared, accelerate))
        from run_experiments import load_jobs
        load_jobs(jobs)
        burn_gate(a.predecessor)
        report = inspect()
        assert report['host'] == a.host and len(report['gpus']) == 8
        assert all('B200' in g['name'] for g in report['gpus'])
        assert digest('/mnt/local/_gpu_guard/gpu_guard.sh') == GUARD_HASH
        assert digest(ROOT/'resources/llm_pretrain_burn.py') == BURN_HASH
        write(a.output/'preflight.json', report)
        run('scripts/bdot_b200_launch.py', '--root', a.output, '--host', a.host,
            '--session', a.session, '--jobs-config', jobs,
            environment={**env, 'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'})
        print('PILOT_QUEUE_SUBMITTED', flush=True)
    except BaseException as error:
        write(a.output/'preparation-failed.json', dict(status='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
