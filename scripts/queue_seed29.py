"""Queue only seed-29 common pretraining after the completed seed-17 workflow."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from pilot_overnight import Workflow, PROJECT, DATA, PREP, ASSETS, PYTHON, ALL, tail, now
from pilot_gpu_ops import check, identity, inspect, preflight, start, stop
from gpu_status import snapshot, require_free

OUT = Path('/mnt/local/_outputs/deep-llms_th2/ccm_pilot_seed29_20260913_a01')
PREVIOUS = Path('/mnt/local/_outputs/deep-llms_th2/ccm_stage2_seed17_20260913_a01')
OLD_SESSION = 'ccm_stage2_20260913_a01'
COMPLETE_EVENT = 'stage2_five_arms_dev_evaluated_and_burns_active'


def train_command():
    return [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
            '--module', 'ccm', 'train', '--phase', 'common', '--arm', 'base', '--seed', '29',
            '--data', str(DATA/'corpus'), '--model-config', str(ASSETS),
            '--output', str(OUT/'common'), '--device', 'cuda', '--activation-checkpointing',
            '--microbatch-segments', '8', '--loss-chunk', '1024', '--save-every', '1000', '--log-every', '10']


def previous_status():
    s = json.loads((PREVIOUS/'status.json').read_text())
    check(s.get('stage') != 'failed' and s.get('success') is not False, 'Previous workflow failed; preserve it')
    age = (datetime.now(timezone.utc)-datetime.fromisoformat(s['time'])).total_seconds()
    check(-10 <= age <= 180, 'Previous heartbeat stale; do not take over')
    return s


def previous_complete():
    path = PREVIOUS/'complete.json'
    if not path.exists():
        return False
    check(not path.is_symlink(), 'Invalid completion path')
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False  # Predecessor may be in the middle of publishing it.
    check(d.get('success') is True and d.get('stage') == 'complete' and d.get('event') == COMPLETE_EVENT,
          'Wrong full-workflow completion marker')
    return True


def observer_identity():
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
        return None  # Exited/zombie pane process cannot restart burns.
    check(b'scripts/pilot_stage2.py' in ident[2], 'Unexpected previous observer')
    return pid, ident


class Seed29(Workflow):
    def __init__(self):
        super().__init__()
        self.reclaim_authorized = False

    def record(self, **fields):
        record = dict(time=now(), stage=self.stage, **fields)
        print(json.dumps(record), flush=True)
        pending = OUT/'status.next.json'
        pending.write_text(json.dumps(record, indent=2)+'\n')
        os.replace(pending, OUT/'status.json')

    def burns(self, indices):
        self.burn_number += 1
        label = f'ccm_seed29_common_20260913_a01_b{self.burn_number}'
        start(indices, label, OUT/f'{label}.log', 30040+self.burn_number)

    def run(self, name, command, indices=(), spare=()):
        self.stage = name
        self.record(event='launch', argv=command, indices=list(indices))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, indices)),
                   CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='offline', PYTHONUNBUFFERED='1',
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTHONPATH=str(PROJECT))
        for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key, None)
        if indices:
            require_free(list(indices))
        with (OUT/f'{name}.log').open('x') as log:
            self.child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
            while True:
                try:
                    code = self.child.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    if spare:
                        self.keep_burns(list(spare))
                    self.record(event='running', pid=self.child.pid, gpus=snapshot(ALL),
                                task_log_tail=tail(OUT/f'{name}.log'))
            self.child = None
        self.record(event='exit', exit_code=code, task_log_tail=tail(OUT/f'{name}.log'))
        check(code == 0, f'{name} failed with exit {code}; no automatic retry')


    def wait_previous(self):
        self.stage = 'waiting_for_full_stage2_completion'
        while True:
            s = previous_status()
            if previous_complete() and s.get('stage') == 'complete' and s.get('success') is True:
                self.record(event='previous_full_workflow_complete', previous_time=s['time'])
                return
            check(observer_identity() is not None, 'Previous workflow ended without complete verification')
            self.record(event='waiting_readonly', previous_stage=s['stage'], previous_time=s['time'])
            time.sleep(30)

    def stable_burn_window(self):
        self.stage = 'verify_three_minutes_of_burns'
        started, deadline, group = None, time.monotonic()+600, None
        while time.monotonic() < deadline:
            s = previous_status()
            check(previous_complete() and s['stage'] == 'complete' and s.get('success') is True,
                  'Previous completion lost')
            result = inspect(ALL, active=True)
            observed = (result[1], result[2], result[3])
            if observed != group:
                started, group = time.monotonic(), observed
            elapsed = time.monotonic()-started
            self.record(event='stable_burn_observation', elapsed_seconds=elapsed,
                        launcher=result[2], gpus=result[0])
            if elapsed >= 180:
                return
            time.sleep(30)
        raise RuntimeError('Burn group did not remain stable for three minutes')

    def disarm_previous(self):
        self.stage = 'disarm_completed_stage2_observer'
        check(previous_complete(), 'Never disarm an incomplete workflow')
        observed = observer_identity()
        if observed is not None:
            flag = PREVIOUS/'STOP_IDLE_WATCH'
            check(not flag.is_symlink(), 'Invalid stop flag')
            if not flag.exists():
                flag.open('x').close()
            pid, expected = observed
            deadline = time.monotonic()+360
            while time.monotonic() < deadline:
                try:
                    alive = identity(pid) == expected
                except (FileNotFoundError, ProcessLookupError):
                    alive = False
                if not alive:
                    break
                self.record(event='waiting_for_observer_exit', pid=pid)
                time.sleep(30)
            else:
                raise RuntimeError('Previous observer did not exit; burns untouched')
        check(observer_identity() is None, 'Previous observer still active or replaced')
        self.record(event='previous_observer_exited')

    def recover_idle(self):
        # The old workflow exclusively owns GPUs until its observer has exited.
        if not self.reclaim_authorized:
            self.record(event='recovery_deferred_to_previous_workflow')
            return
        super().recover_idle()

    def keep_burns(self, indices):
        if not self.reclaim_authorized:
            inspect(indices, active=True)  # Old observer is the only restarter.
        else:
            super().keep_burns(indices)

    def execute(self):
        preflight()
        check(Path.cwd() == PROJECT and os.environ.get('CONDA_DEFAULT_ENV') == 'train_env', 'Wrong project/environment')
        check(not OUT.exists() and not OUT.is_symlink(), 'Seed-29 output must be fresh')
        OUT.mkdir()
        self.owns_output = True
        self.wait_previous()
        self.stable_burn_window()
        self.run('validate_previous', [PYTHON, 'scripts/validate_seed29.py', 'previous', '--workflow', str(OUT)], spare=ALL)
        self.run('validate_data', [PYTHON, 'scripts/validate_pilot_handoff.py', 'data',
                 '--data', str(DATA), '--prep-reports', str(PREP), '--workflow', str(OUT),
                 '--assets', str(ASSETS)], spare=ALL)
        self.disarm_previous()
        self.reclaim_authorized = True
        inspect(ALL, active=True)
        stop(ALL)  # pidfds of freshly verified burn workers only.
        self.free_after_wait(ALL)
        self.free_after_wait(ALL)
        self.run('train_common_seed29', train_command(), ALL)
        self.free_after_wait(ALL)
        self.run('validate_common', [PYTHON, 'scripts/validate_seed29.py', 'common', '--workflow', str(OUT)])
        self.burns(ALL)
        self.stage = 'complete'
        self.record(event='seed29_common_verified_and_burns_active', success=True, gpus=snapshot(ALL))
        with (OUT/'complete.json').open('x') as f:
            f.write((OUT/'status.json').read_text())
        while not (OUT/'STOP_IDLE_WATCH').exists():
            time.sleep(60)
            if (OUT/'STOP_IDLE_WATCH').exists():
                break
            self.keep_burns(ALL)
            self.record(event='complete_burn_health_check', success=True, gpus=snapshot(ALL))


if __name__ == '__main__':
    w = Seed29()
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
