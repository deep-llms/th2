"""Eight owned GPU lanes; failure in one parent does not cancel other parents."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import time
from efficient_llm.a2_depth.common import read, write, Checks, DATASETS, PARENTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--receipt', required=True)
    a = p.parse_args()
    cfg = read(a.config)
    root, started = Path(cfg['results_root']), time.monotonic()
    checks, events = Checks(root/'coordinator_checks.json'), []
    work = [(ds, parent) for ds in DATASETS for parent in PARENTS]
    write(root/'progress.json', dict(phase='part_a', expected_parents=12, expected_pruned_configs=144))

    def lane(gpu):
        results = []
        for ds, parent in work[gpu::8]:
            name = ds+'-'+parent
            log = root/'logs'/(name+'.log')
            log.parent.mkdir(exist_ok=True)
            try:
                with log.open('x') as f:
                    proc = subprocess.run([sys.executable, '-u', '-m', 'efficient_llm.a2_depth.parta',
                        '--config', a.config, '--dataset', ds, '--parent', parent],
                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}, stdout=f, stderr=subprocess.STDOUT)
                event = dict(name=name, returncode=proc.returncode, log=str(log), gpu=gpu)
            except Exception as error:
                event = dict(name=name, returncode=-1, error=repr(error), gpu=gpu)
            results.append(event)
            write(root/'lanes'/f'{gpu}.json', dict(events=results))
        return results

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for results in pool.map(lane, range(8)):
            for event in results:
                events.append(event)
                checks.record('process:'+event['name'], event['returncode'] == 0, event)
    write(root/'ledger.json', dict(part='A', elapsed_seconds=time.monotonic()-started,
          expected_node_hours=.6, policy='report-only', events=events))
    from scripts.a2_depth_parta_report import finalize
    decision = checks.attempt('part_a_readout', lambda: finalize(cfg))
    write(root/'progress.json', dict(phase='part_a_terminal', completed=events,
          next='publish Part A note, then DT4 and Part B in a separately frozen authorized stage'))
    write(a.receipt, dict(execution_complete=True, coverage_complete=bool(decision and decision['coverage_complete']),
          readout_complete=decision is not None, elapsed_seconds=time.monotonic()-started, events=events))


if __name__ == '__main__':
    main()
