"""Bounded synthetic B-DOT verification on already allocated eight B200 GPUs.

The outer run_then_burn supervisor owns GPU reclamation, timeout and restoration.
Independent numerical cases retain failures and continue; the final exit fails
if any required case fails. No corpus download or research pilot is launched.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gpu_status import require_free


def build_cases(out, accelerate_config):
    py = sys.executable
    cases = []
    def add(name, argv, gpus, receipt=None, required=True):
        cases.append(dict(name=name, argv=[str(x) for x in argv], gpus=gpus,
                          receipt=str(receipt) if receipt else None, required=required))
    add('cpu_bdot', [py, '-m', 'unittest', 'discover', '-s', 'tests/bdot', '-p', 'test_*.py', '-v'], [])
    for name, extra in [('native_math', ['--reference-math']), ('native_default', []),
                        ('native_self', ['--native-self-control'])]:
        receipt = out/(name+'.json')
        add(name, [py, 'tests/bdot/gpu_probe.py', '--full-native', '--native-only',
                   *extra, '--output', receipt], [0], receipt)
    receipt = out/'native_fa4.json'
    add('native_fa4', [py, 'tests/ml/fa4_gpu_probe.py', receipt], [0], receipt)
    for stable in (False, True):
        name = 'routed_fa4_stable' if stable else 'routed_fa4_natural'
        receipt = out/(name+'.json')
        add(name, [py, 'tests/bdot/gpu_probe.py', '--routing-only', '--fa4',
                   *(['--stable-router-bias'] if stable else []), '--output', receipt],
            [0], receipt, required=stable)
    for soft in (False, True):
        name = 'ddp_soft' if soft else 'ddp_hard'
        add(name, [py, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                   'tests/bdot/ddp_probe.py', out/name, '--device', 'cuda', '--checkpointing',
                   *(['--soft'] if soft else [])], [0, 1], out/name/'rank0.json')
    for backend in ('sdpa', 'fa4'):
        for label, config, extra in [
            ('native', 'dense_native', []), ('D', 'D', []),
            ('hard_mixed', 'P1-B', ['--progress', '.7', '--router-prior', '0']),
            ('soft_mixed', 'P1-B', ['--progress', '.01', '--router-prior', '0'])]:
            name = f'bench_{backend}_{label}'
            receipt = out/(name+'.json')
            add(name, [py, '-m', 'accelerate.commands.launch', '--config_file', accelerate_config,
                '--main_process_port', '0', 'scripts/benchmark_bdot.py',
                '--device', 'cuda', '--config', f'configs/bdot/{config}.json',
                '--output', receipt, '--batch-size', '16', '--accumulation', '4',
                '--length', '1024', '--document-length', '128', '--bf16',
                '--warmup', '3', '--steps', '10', '--attention-backend', backend, *extra],
                list(range(8)), receipt)
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--accelerate-config', type=Path, required=True)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    out = args.output.resolve()
    cases = build_cases(out, args.accelerate_config.resolve())
    if args.plan_only:
        print(json.dumps(cases, indent=2))
        return 0
    if not args.accelerate_config.is_file():
        raise ValueError('Missing explicit Accelerate configuration')
    out.mkdir(parents=True, exist_ok=False)
    (out/'cases.json').write_text(json.dumps(cases, indent=2)+'\n')
    results = []
    for case in cases:
        require_free(list(range(8)))
        print('VERIFY_START '+case['name'], flush=True)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(map(str, case['gpus'])),
               'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        start = time.time()
        with (out/(case['name']+'.log')).open('x') as log:
            run = subprocess.run(case['argv'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = dict(name=case['name'], returncode=run.returncode,
                      seconds=time.time()-start, required=case['required'])
        if case['receipt'] and Path(case['receipt']).is_file():
            record['receipt'] = json.loads(Path(case['receipt']).read_text())
        # A process exit cannot substitute for the declared numerical receipt.
        good_receipt = not case['receipt'] or record.get('receipt', {}).get('status') in ('passed', 'completed')
        record['passed'] = run.returncode == 0 and good_receipt
        results.append(record)
        (out/'progress.json').write_text(json.dumps(results, indent=2)+'\n')
        print('VERIFY_RESULT '+json.dumps({k:v for k,v in record.items() if k!='receipt'}), flush=True)
        time.sleep(2)
        require_free(list(range(8)))
    ratios = {}
    for backend in ('sdpa', 'fa4'):
        speed = {r['name'].removeprefix(f'bench_{backend}_'): r['receipt']['tokens_per_second']
                 for r in results if r['name'].startswith(f'bench_{backend}_') and r['passed']}
        if {'D', 'native', 'hard_mixed', 'soft_mixed'} <= speed.keys():
            ratios[backend] = dict(tokens_per_second=speed,
                hard_over_D=speed['hard_mixed']/speed['D'],
                hard_over_native=speed['hard_mixed']/speed['native'],
                soft_over_D=speed['soft_mixed']/speed['D'],
                soft_over_native=speed['soft_mixed']/speed['native'],
                hard_runtime_gate=speed['hard_mixed']/speed['native'] >= .6)
    failed = [r['name'] for r in results if r['required'] and not r['passed']]
    summary = dict(status='failed' if failed else 'passed', failed_required=failed,
                   results=results, throughput=ratios,
                   scope='Synthetic numerical checks and complete HF Trainer updates; no research training.')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('BDOT_VERIFICATION_SUMMARY '+json.dumps(dict(status=summary['status'],
          failed_required=failed, throughput=ratios)), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
