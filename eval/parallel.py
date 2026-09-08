"""One subprocess per explicitly selected free GPU; no burn/kill operations."""
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import time


def run(jobs, gpus, output_dir, project_dir):
    from scripts.gpu_status import require_free
    from capacity_allocation.data import write_json
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError('Explicit unique nonnegative physical GPU indices required')
    if not jobs or len({job['name'] for job in jobs}) != len(jobs):
        raise ValueError('Nonempty unique jobs required')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Run the queue directly, not under torchrun/accelerate')
    require_free(gpus)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'plan.json', dict(jobs=jobs, gpus=gpus))
    pending, free, active, completed, failed = deque(jobs), deque(gpus), [], [], []
    try:
        while pending or active:
            while pending and free and not failed:
                gpu = free.popleft()
                require_free([gpu])
                job = pending.popleft()
                log = (root/f'{job["name"]}.log').open('x')
                env = os.environ.copy()
                env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                    env.pop(key, None)
                try:
                    proc = subprocess.Popen(job['argv'], cwd=project_dir, env=env,
                        stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                active.append((job, gpu, proc, log))
                print(f'START {job["name"]} physical_gpu={gpu} pid={proc.pid}', flush=True)
            for entry in list(active):
                job, gpu, proc, log = entry
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                active.remove(entry)
                free.append(gpu)
                error = None
                try:
                    if code != 0:
                        raise RuntimeError(f'Worker exit {code}')
                    result = json.loads(Path(job['result']).read_text())
                    if result.get('success') is not True or any(result.get(k) != v for k, v in job['expected'].items()):
                        raise ValueError('Worker result/provenance mismatch')
                    if result['checkpoint']['path'] != job['checkpoint']:
                        raise ValueError('Checkpoint mismatch')
                    if job['stage'] == 'ppl' and set(result['ppl']['by_language']) != set(job['expected']['languages']):
                        raise ValueError('Missing language PPL')
                    if job['stage'] != 'ppl' and set(result['benchmarks']) != set(job['tasks']):
                        raise ValueError('Missing benchmark results')
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                    error = str(exc)
                record = dict(name=job['name'], result=job['result'], exit_code=code, error=error)
                (failed if error else completed).append(record)
                print(f'DONE {record}', flush=True)
            if failed:
                pending.clear()  # Do not start additional work; existing workers finish naturally.
            if active:
                time.sleep(2)
    finally:
        # No GPU PID signals/process-group kills. If orchestration fails, wait
        # for already-owned workers before reporting. No success marker then.
        for _, _, proc, log in active:
            proc.wait()
            log.close()
    if failed or len(completed) != len(jobs):
        write_json(root/'failed.json', dict(success=False, completed=completed, failed=failed))
        raise RuntimeError('Evaluation queue incomplete; see logs. No success marker written.')
    write_json(root/'complete.json', dict(success=True, completed=completed))


def add_arguments(parser):
    parser.add_argument('--checkpoints', nargs='+', required=True, help='NAME=/absolute/checkpoint ...')
    parser.add_argument('--languages', default='en')
    parser.add_argument('--tokenizer-name')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--gpus', nargs='+', type=int, required=True, help='Physical GPU indices')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    parser.add_argument('--benchmark-batch-size', type=int, default=8)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--dry-run', action='store_true', help='Print job plan; no GPU queries or jobs')


def common_args(args, checkpoint):
    command = ['--checkpoint', checkpoint, '--languages', args.languages,
        '--dataset-root', str(Path(args.dataset_root).resolve()), '--precision', args.precision,
        '--benchmark-batch-size', str(args.benchmark_batch_size)]
    if args.tokenizer_name:
        command += ['--tokenizer-name', str(Path(args.tokenizer_name).resolve())]
    return command
