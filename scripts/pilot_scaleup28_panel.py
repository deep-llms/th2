"""Phase-B panel (plan v3 §23.B): compiler gate -> 1B compilation -> five
Stage-1 arms -> six Stage-2 arms -> dev evaluations -> bootstrap contrasts ->
replication decision, against the verified theta_10B_28L.

One tmux owner. The job list is generated on-node by the test-locked
`ccm scaleup-jobs --queue panel` and executed verbatim (same {python}/{run_dir}
substitution and required-output contracts as run_experiments.py), with burn
management woven around it: 8-GPU jobs run on a cleared node, single-GPU
evaluations keep burns on GPUs 1-7, CPU contrasts/decision keep all-eight
burns. Never launches replication seeds or any D_val evaluation.
"""
import json
import os
from pathlib import Path
import time
import traceback

import pilot_overnight as runtime
from gpu_status import snapshot, require_free
from pilot_gpu_ops import check, identity, inspect, preflight, start, stop
from scaleup28_config import (ALL, A2_EVENT, A2_SCRIPT, CORE, DATA, DATA28, OUT,
                              PANEL_OUT, PANEL_SESSION, PROJECT, PYTHON, SESSION,
                              SPARE, THETA)

runtime.OUT = PANEL_OUT

EXPECTED_JOBS = 48


def generation_command():
    return [PYTHON, '-m', 'ccm', 'scaleup-jobs', '--queue', 'panel',
            '--data', str(DATA28), '--vocabulary', str(DATA/'vocabulary.npz'),
            '--checkpoint', str(THETA), '--gpus', '0', '1', '2', '3', '4', '5', '6', '7',
            '--output', str(PANEL_OUT/'panel.jobs.json')]


def substitute(argv):
    return [a.replace('{python}', PYTHON).replace('{run_dir}', str(PANEL_OUT)) for a in argv]


def job_kind(job):
    gpus = job.get('gpus')
    if gpus and len(gpus) == 8:
        return 'gpu8'
    if gpus == [0]:
        return 'gpu0'
    if '-vs-' in job['name'] or job['name'] == 'replication-decision':
        return 'cpu_long'
    return 'cpu'


def verify_required(job):
    for spec in job.get('required_outputs', ()):
        path = (PANEL_OUT/spec['path']).resolve()
        check(path.is_relative_to(PANEL_OUT.resolve()), f'Output escapes run root: {spec["path"]}')
        check(path.is_file() and path.stat().st_size > 0, f'Missing/empty artifact: {spec["path"]}')
        if 'json_equals' in spec:
            data = json.loads(path.read_text())
            for key, value in spec['json_equals'].items():
                check(isinstance(data, dict) and key in data and data[key] == value,
                      f'Artifact contract failed: {spec["path"]} [{key}]')


class Panel(runtime.Workflow):
    def __init__(self):
        super().__init__()
        self.reclaimed = False
        self.burn_state = 'none'

    def burns(self, indices):
        self.burn_number += 1
        label = f'{PANEL_SESSION}_b{self.burn_number}'
        start(indices, label, PANEL_OUT/f'{label}.log', 31000+self.burn_number)

    def keep_burns(self, indices):
        if self.reclaimed:
            super().keep_burns(indices)
        else:
            inspect(indices, active=True)

    def recover_idle(self):
        if not self.reclaimed:
            self.record(event='failure_preserved_previous_burn_owner')
            return
        super().recover_idle()

    def disarm(self):
        done = json.loads((OUT/'complete.json').read_text())
        status = json.loads((OUT/'status.json').read_text())
        from datetime import datetime, timezone
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(status['time'])).total_seconds()
        check(done.get('success') is True and done.get('event') == A2_EVENT
              and status.get('success') is True and status.get('stage') == 'complete'
              and -10 <= age <= 180, 'a2 completion/heartbeat invalid; burns untouched')
        current = self.observer()
        check(current is not None, 'Expected a2 observer missing; inspect before takeover')
        inspect(ALL, active=True)
        flag = OUT/'STOP_IDLE_WATCH'
        check(not flag.exists() and not flag.is_symlink(), 'a2 observer already disarmed')
        flag.open('x').close()
        pid, expected = current
        for _ in range(12):
            try:
                alive = identity(pid) == expected
            except (FileNotFoundError, ProcessLookupError):
                alive = False
            if not alive:
                break
            self.record(event='waiting_for_previous_observer_exit', pid=pid)
            time.sleep(30)
        check(self.observer() is None, 'a2 observer remains live; burns untouched')
        self.reclaimed = True
        self.record(event='previous_observer_exited_without_process_signal')

    def observer(self):
        import subprocess
        if subprocess.run(['tmux', 'has-session', '-t', SESSION], capture_output=True).returncode:
            return None
        info = subprocess.check_output(['tmux', 'display-message', '-p', '-t', SESSION,
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
        check(A2_SCRIPT in ident[2], 'Unexpected prior observer')
        return pid, ident

    def ensure_burn_state(self, want):
        """Transitions only via full-subset stop then a fresh single-session
        start, so pilot_gpu_ops.inspect never sees mixed burn launchers."""
        have = self.burn_state
        if have == want:
            return
        if have == 'all':
            stop(ALL)
            self.free_after_wait(ALL)
        elif have == 'spare':
            stop(SPARE)
            self.free_after_wait(ALL)
        if want == 'none':
            self.free_after_wait(ALL)
        elif want == 'all':
            self.burns(ALL)
        else:
            self.burns(SPARE)
        self.burn_state = want

    def run_panel_job(self, job):
        argv = substitute(job['argv'])
        kind = job_kind(job)
        if kind == 'gpu8':
            self.ensure_burn_state('none')
            self.run(job['name'], argv, ALL)
            self.free_after_wait(ALL)
        elif kind == 'gpu0':
            self.ensure_burn_state('spare')
            self.run(job['name'], argv, [0], SPARE)
            self.free_after_wait([0])
        elif kind == 'cpu_long':
            self.ensure_burn_state('all')
            self.run(job['name'], argv, spare=ALL)
        else:
            spare = dict(none=(), spare=SPARE, all=ALL)[self.burn_state]
            self.run(job['name'], argv, spare=spare)
        verify_required(job)

    def execute(self):
        preflight()
        check(Path.cwd() == PROJECT and os.environ.get('CONDA_DEFAULT_ENV') == 'train_env',
              'Wrong project/environment')
        check(not PANEL_OUT.exists() and not PANEL_OUT.is_symlink(), 'Panel output must be fresh')
        PANEL_OUT.mkdir()
        self.owns_output = True
        self.run('preflight', [PYTHON, '-c',
            "from ccm.cli import code_hash; import transformers, torch; "
            f"assert code_hash() == '{CORE}'; "
            "assert transformers.__version__ == '5.9.0'; "
            "print('SCALEUP28_PANEL_CORE_AND_ENV_PREFLIGHT_PASS', torch.__version__, flush=True)"])
        self.run('generate_jobs', generation_command())
        jobs = json.loads((PANEL_OUT/'panel.jobs.json').read_text())['jobs']
        check(len(jobs) == EXPECTED_JOBS and jobs[0]['name'] == 'verify-common-input'
              and jobs[-1]['name'] == 'replication-decision', 'Unexpected panel job manifest')
        self.run('validate_panel_inputs', [PYTHON, 'scripts/validate_scaleup28_panel.py'])
        self.disarm()
        stop(ALL)
        self.free_after_wait(ALL)
        self.free_after_wait(ALL)
        self.burn_state = 'none'
        for job in jobs:
            self.run_panel_job(job)
        self.ensure_burn_state('all')
        inspect(ALL, active=True)
        decision = json.loads((PANEL_OUT/'decision.json').read_text())
        self.stage = 'complete'
        self.record(event='scaleup28_panel_verified_and_burns_active', success=True,
                    decision=decision, gpus=snapshot(ALL))
        with (PANEL_OUT/'complete.json').open('x') as f:
            f.write((PANEL_OUT/'status.json').read_text())
        while not (PANEL_OUT/'STOP_IDLE_WATCH').exists():
            time.sleep(60)
            if (PANEL_OUT/'STOP_IDLE_WATCH').exists():
                break
            self.keep_burns(ALL)
            self.record(event='complete_burn_health_check', success=True, gpus=snapshot(ALL))


if __name__ == '__main__':
    w = Panel()
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
