"""Authorized seed-17 Stage-1 sequence; no Stage-2 or locked-val launch."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from pilot_overnight import Workflow, PROJECT, DATA, PREP, ASSETS, PYTHON, CORE_SHA, ALL, SPARE, tail, now
from pilot_gpu_ops import check, identity, inspect, preflight, start, stop
from gpu_status import snapshot, require_free

COMMON = Path('/mnt/local/_outputs/deep-llms_th2/ccm_pilot_seed17_20260913_a01')
OUT = Path('/mnt/local/_outputs/deep-llms_th2/ccm_stage1_seed17_20260913_a01')
ARMS = ('contextual', 'isolated', 'shuffled', 'shallow', 'delta', 'grad')
OLD_SESSION = 'ccm_pilot_overnight_20260913_a01'


def train_command(arm):
    check(arm in ARMS, 'Unknown Stage-1 arm')
    cmd = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
           '--module', 'ccm', 'train', '--phase', 'stage1', '--arm', arm, '--seed', '17',
           '--data', str(DATA/'corpus'), '--vocabulary', str(DATA/'vocabulary.npz'),
           '--coverage', str(PREP/'coverage.json'), '--checkpoint', str(COMMON/'common/checkpoint-15259'),
           '--output', str(OUT/'train'/arm), '--device', 'cuda', '--activation-checkpointing',
           '--microbatch-segments', '8', '--loss-chunk', '1024', '--save-every', '1000', '--log-every', '10']
    if arm != 'grad':
        cmd += ['--table', str(COMMON/'tables'/arm)]
    return cmd


def eval_command(arm):
    check(arm in ('base',)+ARMS, 'Unknown evaluation arm')
    ck = COMMON/'common/checkpoint-15259' if arm == 'base' else OUT/'train'/arm/'checkpoint-977'
    return [PYTHON, '-m', 'ccm', 'evaluate', '--data', str(DATA/'corpus'),
            '--vocabulary', str(DATA/'vocabulary.npz'), '--checkpoint', str(ck),
            '--output', str(OUT/'eval'/arm), '--role', 'dev', '--device', 'cuda',
            '--microbatch-segments', '8', '--loss-chunk', '1024',
            '--diagnostic-table', str(COMMON/'tables/contextual')]


class Stage1(Workflow):
    # Reuse the previously tested wait/free, burn ownership and failure helpers.
    def record(self, **fields):
        record = dict(time=now(), stage=self.stage, **fields)
        print(json.dumps(record), flush=True)
        pending = OUT/'status.next.json'
        pending.write_text(json.dumps(record, indent=2)+'\n')
        os.replace(pending, OUT/'status.json')

    def burns(self, indices):
        self.burn_number += 1
        label = f'ccm_stage1_20260913_a01_b{self.burn_number}'
        start(indices, label, OUT/f'{label}.log', 29840+self.burn_number)

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

    def validate(self, stage, arm=None, spare=()):
        cmd = [PYTHON, 'scripts/validate_stage1.py', stage, '--workflow', str(OUT)]
        if arm:
            cmd += ['--arm', arm]
        self.run('validate_'+stage+('_'+arm if arm else ''), cmd, spare=spare)

    def disarm_old_observer(self):
        self.stage = 'disarm_old_idle_observer'
        done = json.loads((COMMON/'complete.json').read_text())
        check(done.get('success') is True and done.get('stage') == 'complete', 'Previous workflow incomplete')
        exists = subprocess.run(['tmux', 'has-session', '-t', OLD_SESSION], capture_output=True).returncode == 0
        if exists:
            info = subprocess.check_output(['tmux', 'display-message', '-p', '-t', OLD_SESSION,
                                            '#{pane_dead} #{pane_pid}'], text=True).split()
            if info[0] == '0':
                pid = int(info[1])
                original = identity(pid)
                check(b'scripts/pilot_overnight.py' in original[2], 'Unexpected previous observer process')
                flag = COMMON/'STOP_IDLE_WATCH'
                check(not flag.is_symlink(), 'Invalid observer stop flag')
                if not flag.exists():
                    flag.open('x').close()
                deadline = time.monotonic()+360
                while time.monotonic() < deadline:
                    try:
                        if identity(pid) != original:
                            break
                    except (FileNotFoundError, ProcessLookupError):
                        break
                    self.record(event='waiting_for_observer_exit', pid=pid, gpus=snapshot(ALL))
                    time.sleep(30)
                else:
                    raise RuntimeError('Previous observer did not exit; burns remain untouched')
        self.record(event='old_observer_disarmed')

    def execute(self):
        preflight()
        check(Path.cwd() == PROJECT and os.environ.get('CONDA_DEFAULT_ENV') == 'train_env', 'Wrong project/environment')
        check(not OUT.exists() and not OUT.is_symlink(), 'Stage-1 output must be fresh')
        OUT.mkdir()
        self.owns_output = True
        (OUT/'train').mkdir()
        (OUT/'eval').mkdir()
        (OUT/'reports').mkdir()
        self.validate('inputs', spare=ALL)  # Keep existing burns through CPU checks.
        self.disarm_old_observer()
        inspect(ALL, active=True)
        stop(ALL)  # Verified worker pidfds only; never name/group signals.
        self.free_after_wait(ALL)
        self.free_after_wait(ALL)
        # This proven trainer uses torchrun directly, not Accelerate.
        for arm in ARMS:
            self.run('train_'+arm, train_command(arm), ALL)
            self.free_after_wait(ALL)
            self.validate('train', arm)
        self.burns(SPARE)
        for arm in ('base',)+ARMS:
            self.run('eval_'+arm, eval_command(arm), [0], SPARE)
            self.free_after_wait([0])
            self.validate('eval', arm, spare=SPARE)
        stop(SPARE)
        self.free_after_wait(ALL)
        self.burns(ALL)
        for right in ('isolated', 'shuffled'):
            self.run('compare_contextual_'+right, [PYTHON, '-m', 'ccm', 'compare',
                     '--left', str(OUT/'eval/contextual'), '--right', str(OUT/'eval'/right),
                     '--population', 'overall', '--cluster', 'doc_id', '--cross-seed-coupling', 'independent',
                     '--replicates', '10000', '--output', str(OUT/'reports'/f'contextual_vs_{right}.json')], spare=ALL)
        self.validate('panel', spare=ALL)
        inspect(ALL, active=True)
        self.stage = 'complete'
        self.record(event='stage1_six_arms_dev_evaluated_and_burns_active', success=True, gpus=snapshot(ALL))
        with (OUT/'complete.json').open('x') as f:
            f.write((OUT/'status.json').read_text())
        # Future reclaim must disarm THIS observer using this new run root.
        while not (OUT/'STOP_IDLE_WATCH').exists():
            time.sleep(60)
            if (OUT/'STOP_IDLE_WATCH').exists():
                break
            self.keep_burns(ALL)
            self.record(event='complete_burn_health_check', success=True, gpus=snapshot(ALL))


if __name__ == '__main__':
    w = Stage1()
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
