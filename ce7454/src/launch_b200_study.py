"""Authorized q359 launch: immutable code, CPU data prep, verified burn handoff."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

VISION = '/mnt/local/conda-py311/envs/ce7454_env/bin/python'
SUPERVISOR = '/mnt/local/conda-py311/envs/eval_fa4/bin/python'
HOST = 'thiennh-p6-q359-worker-0'
RAW = Path('/mnt/local/_data/deep-llms_th2/ce7454/raw')
CACHE = RAW.parent / 'cache'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    project = Path.cwd().resolve()
    sys.path.insert(0, str(project))
    from scripts.verified_gpu_reclaim import inspect
    from scripts.train_then_burn import approved_launcher, APPROVED_BURNS, BURN_HASH
    if project != Path('/mnt/local/deep-llms_th2') or socket.gethostname() != HOST:
        raise RuntimeError('Wrong project/machine')
    before = inspect()
    if before['guard_disabled']:
        raise RuntimeError('Another GPU workflow holds the guard')
    approved = {**APPROVED_BURNS, str(project / 'resources/llm_pretrain_burn.py'): BURN_HASH}
    for pid in before['workers']:
        parent = before['processes'][str(pid)]['ppid']
        if not (approved_launcher(pid, approved) or parent > 1 and approved_launcher(parent, approved)):
            raise RuntimeError(f'Unknown GPU workload {pid}; leaving it untouched')
    args.output.mkdir(parents=True, exist_ok=False)
    code = args.output / 'code'
    source = Path(__file__).resolve().parent
    manifest = json.loads((source.parent / 'source_manifest.json').read_text())
    code.mkdir()
    for name, expected in manifest.items():
        src = source / name
        if hashlib.sha256(src.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Source differs from manifest: {name}')
        shutil.copy2(src, code / name)
    (args.output / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    subprocess.run([VISION, '-m', 'unittest', 'discover', '-s', str(code), '-p', 'test_deadline*.py', '-v'],
                   check=True, env=env)
    if CACHE.exists():
        raise FileExistsError('Expected a fresh CE7454 cache; inspect before reusing')
    subprocess.run([VISION, '-u', str(code / 'prepare_b200_data.py'), '--raw', str(RAW),
                    '--cache', str(CACHE), '--workers', '16'], check=True, env=env)
    fresh = inspect()
    (args.output / 'inspection.json').write_text(json.dumps(fresh, indent=2) + '\n')
    jobs = {'jobs': [{'name': 'ce7454-study',
        'argv': [VISION, '-u', str(code / 'deadline_study.py'), '--cache', str(CACHE),
                 '--output', '{run_dir}/study'], 'gpus': list(range(8)),
        'timeout_seconds': 14 * 3600,
        'required_outputs': [{'path': 'study/complete.json', 'json_equals': {'status': 'passed'}}]}]}
    (args.output / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    print('CPU_PREFLIGHT_PASSED', datetime.now(timezone.utc).isoformat(), flush=True)
    subprocess.run([SUPERVISOR, '-u', '-m', 'scripts.train_then_burn', '--config', str(args.output / 'jobs.json'),
                    '--output', str(args.output / 'supervised'), '--inspection', str(args.output / 'inspection.json'),
                    '--host', HOST, '--burn-session', 'ce7454_final_burn_20261016_a01'], check=True)


if __name__ == '__main__':
    main()
