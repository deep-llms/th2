"""One approved A3 panel, then original all-eight-GPU burns; offline only."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import pilot_overnight as runtime
from pilot_gpu_ops import check, identity, inspect, preflight, start, stop
from gpu_status import snapshot, require_free

OUT = Path('/mnt/local/_outputs/deep-llms_th2/ccm_a3_seed17_20260914_a01')
PREVIOUS = Path('/mnt/local/_outputs/deep-llms_th2/ccm_memory_diagnostics_seed17_20260914_a01')
STAGE2 = Path('/mnt/local/_outputs/deep-llms_th2/ccm_stage2_seed17_20260913_a01')
OLD_SESSION = 'ccm_memory_diagnostics_20260914_a01'
ACTIVE, SPARE, ALL = [0, 1, 2, 3, 4], [5, 6, 7], list(range(8))
ARMS = ('contextual', 'isolated', 'shuffled', 'grad', 'base')
runtime.OUT = OUT  # The existing single-child monitor writes only this fresh workflow.


def command(action, output):
    return [runtime.PYTHON, 'scripts/frequency_variance_diagnostics.py', action,
            '--data', str(runtime.DATA), '--stage2', str(STAGE2), '--output', str(output), '--definition', str(OUT/'definition'),
            '--table', str(runtime.OUT.parent/'ccm_pilot_seed17_20260913_a01/tables/contextual')]


def panel(smoke):
    """Reap every owned child before returning, including after launch failure.

    No cancellation signals. The outer monitor won't restore burns while a
    dispatcher/worker is live or an unknown process still owns any GPU.
    """
    children = []
    errors = []
    try:
        for gpu, arm in zip(ACTIVE, ARMS):
            out = OUT/('smoke' if smoke else 'full')/arm
            argv = command('evaluate', out)+['--arm', arm]
            if smoke:
                argv += ['--max-batches', '2']
            require_free([gpu])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            log = (OUT/f'{"smoke" if smoke else "full"}_{arm}.log').open('x')
            try:
                child = subprocess.Popen(argv, cwd=runtime.PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
                children.append((arm, child))
            finally:
                log.close()
    finally:
        for arm, child in children:
            code = child.wait()
            print(json.dumps(dict(arm=arm, exit_code=code, smoke=smoke)), flush=True)
            if code:
                errors.append((arm, code))
    check(not errors and len(children) == 5, f'Panel failed: {errors}')
    for arm in ARMS:
        root = OUT/('smoke' if smoke else 'full')/arm
        d = json.loads((root/'diagnostics.json').read_text())
        check(d['smoke'] == smoke and d['arm'] == arm, 'Wrong panel artifact')
    print('FIVE_DIAGNOSTIC_CHILDREN_VERIFIED_AND_REAPED', flush=True)


def observer():
    if subprocess.run(['tmux', 'has-session', '-t', OLD_SESSION], capture_output=True).returncode:
        return None
    info = subprocess.check_output(['tmux', 'display-message', '-p', '-t', OLD_SESSION,
                                    '#{pane_dead} #{pane_pid}'], text=True).split()
    if info[0] == '1':
        return None
    pid = int(info[1])
    try:
        ident = identity(pid)
    except (FileNotFoundError, ProcessLookupError):
        return None
    if not any(ident[2]):
        return None
    check(b'scripts/pilot_memory_diagnostics.py' in ident[2], 'Unexpected current observer; no takeover')
    return pid, ident


class Diagnostics(runtime.Workflow):
    def __init__(self):
        super().__init__()
        self.reclaimed = False

    def burns(self, indices):
        self.burn_number += 1
        name = f'ccm_a3_20260914_a01_b{self.burn_number}'
        start(indices, name, OUT/f'{name}.log', 30240+self.burn_number)

    def keep_burns(self, indices):
        if self.reclaimed:
            super().keep_burns(indices)
        else:
            inspect(indices, active=True)

    def disarm(self):
        complete = json.loads((PREVIOUS/'complete.json').read_text())
        status = json.loads((PREVIOUS/'status.json').read_text())
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(status['time'])).total_seconds()
        check(complete.get('success') is True and complete.get('event') == 'memory_diagnostics_verified_and_burns_active'
              and status.get('success') is True and status['stage'] == 'complete' and -10 <= age <= 180,
              'A1/A2 completion/heartbeat not valid; burns untouched')
        observed = observer()
        check(observed is not None, 'Expected live completed observer before takeover')
        inspect(ALL, active=True)
        flag = PREVIOUS/'STOP_IDLE_WATCH'
        check(not flag.exists() and not flag.is_symlink(), 'Observer was already disarmed; inspect manually')
        flag.open('x').close()
        pid, expected = observed
        for _ in range(12):
            try:
                alive = identity(pid) == expected
            except (FileNotFoundError, ProcessLookupError):
                alive = False
            if not alive:
                break
            self.record(event='waiting_for_previous_observer_exit', pid=pid)
            time.sleep(30)
        check(observer() is None, 'Previous observer still running; burns untouched')
        self.reclaimed = True
        self.record(event='previous_observer_exited_no_process_signal')

    def recover_idle(self):
        if not self.reclaimed or (self.child is not None and self.child.poll() is None):
            self.record(event='recovery_refused_previous_owner_or_live_child')
            return
        gpus = snapshot(ALL)
        if all(not g['pids'] for g in gpus):
            self.burns(ALL)
        elif all(not gpus[i]['pids'] for i in ACTIVE) and all(gpus[i]['pids'] for i in SPARE):
            inspect(SPARE)
            stop(SPARE)
            self.free_after_wait(ALL)
            self.burns(ALL)
        else:
            self.record(event='existing_workloads_preserved', gpus=gpus)

    def execute(self):
        preflight()
        check(Path.cwd() == runtime.PROJECT and os.environ.get('CONDA_DEFAULT_ENV') == 'train_env',
              'Wrong project/environment')
        check(not OUT.exists() and not OUT.is_symlink(), 'Diagnostic output must be fresh')
        OUT.mkdir()
        self.owns_output = True
        self.run('preflight', command('preflight', OUT), spare=ALL)
        self.disarm()
        stop(ALL)
        self.free_after_wait(ALL)
        self.free_after_wait(ALL)
        self.burns(SPARE)
        self.run('smoke', ['/usr/bin/python3', 'scripts/pilot_a3_diagnostics.py', 'smoke'], ACTIVE, SPARE)
        require_free(ACTIVE)
        self.run('full_dev', ['/usr/bin/python3', 'scripts/pilot_a3_diagnostics.py', 'full'], ACTIVE, SPARE)
        self.free_after_wait(ACTIVE)
        stop(SPARE)
        self.free_after_wait(ALL)
        self.burns(ALL)
        # Document bootstrap is CPU-only; all eight burns are already restored.
        self.run('summarize', command('summarize', OUT/'full'), spare=ALL)
        self.run('pack', command('pack', OUT), spare=ALL)
        summary = json.loads((OUT/'full/summary.json').read_text())
        check(set(summary['arms']) == set(ARMS), 'Incomplete summary')
        inspect(ALL, active=True)
        self.stage = 'complete'
        self.record(event='a3_diagnostics_verified_and_burns_active', success=True, gpus=snapshot(ALL))
        (OUT/'complete.json').write_text((OUT/'status.json').read_text())
        while not (OUT/'STOP_IDLE_WATCH').exists():
            time.sleep(60)
            if (OUT/'STOP_IDLE_WATCH').exists():
                break
            self.keep_burns(ALL)
            self.record(event='complete_burn_health_check', success=True, gpus=snapshot(ALL))


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in ('smoke', 'full'):
        panel(sys.argv[1] == 'smoke')
    else:
        check(len(sys.argv) == 1, 'Unexpected workflow arguments')
        w = Diagnostics()
        try:
            w.execute()
        except Exception as error:
            traceback.print_exc()
            if w.owns_output:
                w.stage = 'failed'
                w.record(event='failure', error=str(error), success=False)
                try:
                    w.recover_idle()
                except Exception:
                    traceback.print_exc()
            raise SystemExit(1)
