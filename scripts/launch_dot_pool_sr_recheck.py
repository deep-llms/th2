"""Admit two evaluation-only corrections through the existing native supervisor."""
import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from efficient_llm.dot_pool.data import sha, write
from run_experiments import load_jobs
from scripts.dot_pool_report_recovery import check_burn
from scripts.dot_pool_sr_recheck import check_inputs
from scripts.train_then_burn import BURN_HASH, GUARD_HASH

BASE = Path('/mnt/local/_outputs/deep-llms_th2/dot_pool_20261112')
HOST = 'thiennh-p6-wc69-worker-0'


def job_config(source):
    return {'jobs': [
        {'name': f'E1-SR-s{seed}-diagnostics', 'gpus': [seed],
         'argv': ['{python}', '-u', '-m', 'scripts.dot_pool_sr_recheck',
                  '--expected', str(Path(source) / f'expected-s{seed}.json'),
                  '--output', '{run_dir}/E1-SR-s' + str(seed)],
         'required_outputs': [{'path': f'E1-SR-s{seed}/complete.json',
                               'json_equals': {'verified': True, 'training_performed': False}}]}
        for seed in [0, 1]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    target = BASE / 'sr-recheck-a01'
    session = 'dot_pool_sr_recheck_20261112_a01'
    if socket.gethostname() != HOST or len(args.commit) != 40 or any(c not in '0123456789abcdef' for c in args.commit):
        raise ValueError('Node/source identity differs')
    if target.exists() or subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
        raise ValueError('Refuse duplicate correction')
    pin = json.loads((source / 'resources/dot_pool_sr_recheck_20261112.json').read_text())
    for rel, value in pin['source_files'].items():
        if sha(source / rel) != value:
            raise ValueError('Pinned correction source differs: ' + rel)
    for package, wanted in [('torch', '2.14.1'), ('flash-attn-4', '4.0.0b33')]:
        if version(package).split('+')[0] != wanted:
            raise ValueError('Evaluation runtime differs: ' + package)
    predecessor = BASE / 'execution-a03/source/resources/llm_pretrain_burn.py'
    if sha(predecessor) != BURN_HASH or sha('/mnt/local/_gpu_guard/gpu_guard.sh') != GUARD_HASH:
        raise ValueError('Native handoff source differs')
    for item in pin['runs']:
        check_inputs(item)
    # Read-only exact PID/start-time/command/UUID check, before any reclamation.
    before = check_burn(pin['burn'])
    target.mkdir()
    frozen = target / 'source'
    files = list(source.glob('*.py'))
    for folder in ['efficient_llm', 'scripts']:
        files.extend((source / folder).rglob('*.py'))
    files.append(source / 'resources/llm_pretrain_burn.py')
    hashes = {}
    for original in sorted(set(files)):
        rel = original.relative_to(source);dest = frozen / rel
        dest.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(original, dest)
        hashes[str(rel)] = sha(dest)
    for rel, value in pin['source_files'].items():
        if sha(frozen / rel) != value:
            raise ValueError('Frozen correction source differs: ' + rel)
    for seed, item in enumerate(pin['runs']):
        write(frozen / f'expected-s{seed}.json', item)
    write(target / 'source-manifest.json', {'commit': args.commit, 'files': hashes})
    write(target / 'admission.json', {'inputs_verified': True, 'runs': pin['runs'], 'training_performed': False})
    # Inspect again after copying. The unchanged supervisor rechecks before signaling.
    inspection = check_burn(pin['burn'])
    write(target / 'inspection.json', inspection)
    jobs = target / 'jobs.json';jobs.write_text(json.dumps(job_config(frozen), indent=2) + '\n')
    load_jobs(jobs)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
           'PYTHONPATH': str(frozen), 'OMP_NUM_THREADS': '2', 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
           'HF_HUB_DISABLE_TELEMETRY': '1', 'WANDB_MODE': 'offline', 'NCCL_NVLS_ENABLE': '0'}
    argv = [sys.executable, '-u', '-m', 'scripts.train_then_burn', '--config', str(jobs),
            '--output', str(target / 'supervised'), '--inspection', str(target / 'inspection.json'),
            '--host', HOST, '--burn-session', session + '_burn',
            '--predecessor-burn-script', str(predecessor)]
    command = 'exec ' + shlex.join(argv) + ' >' + shlex.quote(str(target / 'supervisor.log')) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(frozen), command], env=env, check=True)
    subprocess.run(['tmux', 'set-option', '-w', '-t', session, 'remain-on-exit', 'on'], check=True)
    write(target / 'submitted.json', {'commit': args.commit, 'session': session, 'evaluation_only': True})
    print('DOT_SR_CORRECTION_SUBMITTED', str(target), flush=True)


if __name__ == '__main__':
    main()
