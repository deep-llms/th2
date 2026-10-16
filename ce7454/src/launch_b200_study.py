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
    p.add_argument('--reuse-verified-cache', action='store_true')
    p.add_argument('--previous-output', type=Path)
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
    if args.previous_output:
        previous = json.loads((args.previous_output / 'supervised/supervisor.json').read_text())
        burn = previous.get('burn', {})
        if not burn.get('collective_progress_verified') or 'finished_at' not in previous:
            raise RuntimeError('Previous supervisor has not verified final burn handoff')
        from scripts.verified_gpu_reclaim import ownership
        if ownership(burn['gpus']) != ownership(before['gpus']):
            raise RuntimeError('Current GPU identities differ from verified final burns')
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
        if not args.reuse_verified_cache:
            raise FileExistsError('Existing cache requires explicit verified reuse')
        check = '''
import hashlib, json, numpy as np
from pathlib import Path
raw=Path('/mnt/local/_data/deep-llms_th2/ce7454/raw')
cache=raw.parent/'cache'
expected='ef57baacee12b92038a845308b0911dd71c5b564bdda43b516ed2f90fbb08ce3'
with (raw/'train.zip').open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==expected
ready=json.loads((cache/'ready.json').read_text())
assert ready['archive_sha256']==expected and ready['images']==15000
manifest=json.loads((raw/'manifest.json').read_text())
meta=json.loads((cache/'train_meta.json').read_text())
assert meta=={'filenames':[r['filename'] for r in manifest['images']], 'labels':[r['label'] for r in manifest['images']]}
a=np.load(cache/'train_images.npy',mmap_mode='r')
assert a.shape==(15000,512,512,3) and a.dtype==np.uint8
with (cache/'train_images.npy').open('rb') as f: print('REUSED_CACHE_SHA256',hashlib.file_digest(f,'sha256').hexdigest(),flush=True)
'''
        subprocess.run([VISION, '-c', check], check=True, env=env)
    else:
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
                    '--host', HOST, '--burn-session', args.output.name + '_final_burn'], check=True)


if __name__ == '__main__':
    main()
