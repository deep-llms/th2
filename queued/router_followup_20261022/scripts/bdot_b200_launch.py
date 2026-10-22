"""Launch the authorized bounded verification through the established q359 supervisor."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--session', required=True)
    p.add_argument('--host', default='thiennh-p6-q359-worker-0',
                   help='Explicit authorized node; must match the fresh preflight receipt')
    p.add_argument('--verify-python', type=Path,
                   help='Also verify Accelerate in the selected training interpreter')
    p.add_argument('--jobs-config', type=Path, default=ROOT/'resources/bdot_b200_verification.json')
    p.add_argument('--predecessor-burn-script', type=Path,
                   help='Exact verified enhanced-burn script in a preceding source snapshot')
    args = p.parse_args()
    root = args.root.resolve()
    inspection = root/'preflight.json'
    old = json.loads(inspection.read_text())
    if socket.gethostname() != old['host'] or old['host'] != args.host:
        raise ValueError('Node identity differs from preflight')
    if (root/'supervised').exists() or (root/'supervisor.log').exists():
        raise ValueError('Choose a fresh verification output')
    expected = {'torch':'2.14.1', 'transformers':'5.9.0', 'accelerate':'1.13.0',
                'datasets':'4.8.5', 'tokenizers':'0.22.2', 'safetensors':'0.8.0',
                'flash-attn-4':'4.0.0b33'}
    actual = {name: version(name) for name in expected}
    if any(actual[k].split('+')[0] != v for k, v in expected.items()):
        raise ValueError('Verification runtime differs from pins: '+str(actual))
    from packed_fa4 import register_backend
    register_backend()  # Import/API check only; no tensor allocation before reclaim.
    from accelerate.commands.config.config_args import default_config_file
    import yaml
    source = ROOT/'resources/accelerate_b200.yaml'
    target = Path(default_config_file)
    config = yaml.safe_load(source.read_text())
    if (config['num_processes'] != 8 or config['distributed_type'] != 'MULTI_GPU'
            or config['mixed_precision'] != 'bf16' or config['use_cpu']):
        raise ValueError('Unexpected distributed config')
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = root/'accelerate_before.yaml'
    if backup.exists():
        raise ValueError('Previous launch preparation exists')
    if target.exists():
        shutil.copy2(target, backup)
    shutil.copy2(source, target)
    assert target.read_bytes() == source.read_bytes()
    result = subprocess.run([sys.executable, '-m', 'accelerate.commands.env'],
                            text=True, capture_output=True, check=True)
    (root/'accelerate_env.txt').write_text(result.stdout+result.stderr)
    print('ACCELERATE_CONFIG_VERIFIED '+str(target), flush=True)
    print(result.stdout, flush=True)
    if args.verify_python:
        if not args.verify_python.is_file():
            raise ValueError('Selected training interpreter is missing')
        selected = subprocess.run([str(args.verify_python), '-m', 'accelerate.commands.env'],
                                  text=True, capture_output=True, check=True)
        (root/'selected_accelerate_env.txt').write_text(selected.stdout+selected.stderr)
        print('SELECTED_ACCELERATE_ENV '+selected.stdout, flush=True)
    from run_experiments import load_jobs
    jobs_config = args.jobs_config.resolve()
    load_jobs(jobs_config)
    if subprocess.run(['tmux','has-session','-t',args.session], capture_output=True).returncode == 0:
        raise ValueError('Session already exists')
    environment = dict(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='offline', NCCL_NVLS_ENABLE='0',
        OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    argv = ['env', *(k+'='+v for k,v in environment.items()), sys.executable,
            '-m','scripts.train_then_burn','--config',str(jobs_config),
            '--output',str(root/'supervised'),'--inspection',str(inspection),
            '--host',old['host'],'--burn-session',args.session+'_burn']
    if args.predecessor_burn_script:
        from scripts.train_then_burn import digest, BURN_HASH
        if digest(args.predecessor_burn_script) != BURN_HASH:
            raise ValueError('Predecessor burn script differs from approved source')
        argv += ['--predecessor-burn-script', str(args.predecessor_burn_script.resolve())]
    command = 'cd '+shlex.quote(str(ROOT))+' && exec '+shlex.join(argv)
    command += ' >'+shlex.quote(str(root/'supervisor.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',args.session,command],check=True)
    subprocess.run(['tmux','set-option','-w','-t',args.session,'remain-on-exit','on'],check=True)
    receipt = dict(status='submitted', session=args.session, root=str(root),
                   python=sys.executable, versions=actual, accelerate_cache=str(target),
                   environment=environment, jobs_config=str(jobs_config))
    with (root/'launch.json').open('x') as handle:
        json.dump(receipt,handle,indent=2)
    print('BDOT_VERIFICATION_SUBMITTED '+json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
