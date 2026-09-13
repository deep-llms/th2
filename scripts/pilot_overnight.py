"""Run-specific offline Step 1 -> common training -> table compilation handoff.

Run in persistent tmux with system Python (pidfd support). Research children use
the explicitly activated train_env. Never retries training or deletes output.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from gpu_status import snapshot, require_free
from pilot_gpu_ops import check, identity, inspect, preflight, start, stop

PROJECT = Path('/mnt/local/deep-llms_th2')
DATA = Path('/mnt/local/_data/deep-llms_th2/ccm/pilot_v1_20260913_a01')
PREP = Path('/mnt/local/_outputs/deep-llms_th2/ccm_prepare_pilot_v1_20260913_a01')
OUT = Path('/mnt/local/_outputs/deep-llms_th2/ccm_pilot_seed17_20260913_a01')
ASSETS = Path('/mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base-da87bfb608c14b7cf20ba1ce41287e8de496c0cd')
PYTHON = '/mnt/local/conda-py311/envs/train_env/bin/python'
CORE_SHA = '055f0518853e76487a4e2f31b81f1441113d4fceab322b5e211359a7d58f2fa9'
ALL = list(range(8))
SPARE = list(range(1, 8))
PREP_PID = 19093
PREP_START = '190112375'
PREP_MARKER = 'CCM_PILOT_STEP1_COMPLETE_NO_TRAINING_LAUNCHED'


def now():
    return datetime.now(timezone.utc).isoformat()


def tail(path, size=4096):
    try:
        with Path(path).open('rb') as f:
            f.seek(max(0, f.seek(0, 2)-size))
            return f.read().decode(errors='replace')
    except FileNotFoundError:
        return ''


def prep_alive():
    try:
        ident = identity(PREP_PID)
        return ident[1] == PREP_START and b'scripts/prepare_pilot_data.sh' in ident[2]
    except (FileNotFoundError, ProcessLookupError):
        return False


def prep_finished(log):
    # Completion JSON alone is insufficient: require the shell's final marker
    # and that the exact preparation shell has ended, then validate artifacts.
    return (not prep_alive() and (PREP/'complete.json').is_file()
            and (PREP/'gpus_after.json').is_file() and PREP_MARKER in tail(log, 32768))


def train_command():
    return [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
            '--module', 'ccm', 'train', '--data', str(DATA/'corpus'),
            '--output', str(OUT/'common'), '--phase', 'common', '--arm', 'base',
            '--seed', '17', '--model-config', str(ASSETS), '--device', 'cuda',
            '--activation-checkpointing', '--microbatch-segments', '8',
            '--loss-chunk', '1024', '--save-every', '1000', '--log-every', '10']


def compile_command():
    return [PYTHON, '-m', 'ccm', 'compile', '--data', str(DATA/'corpus'),
            '--vocabulary', str(DATA/'vocabulary.npz'), '--checkpoint',
            str(OUT/'common/checkpoint-15259'), '--output', str(OUT/'tables'),
            '--device', 'cuda', '--microbatch-segments', '4', '--isolated-batch', '256']


class Workflow:
    def __init__(self):
        self.stage = 'starting'
        self.burn_number = 0
        self.child = None
        self.owns_output = False

    def record(self, **fields):
        record = dict(time=now(), stage=self.stage, **fields)
        print(json.dumps(record), flush=True)
        pending = OUT/'status.next.json'
        pending.write_text(json.dumps(record, indent=2)+'\n')
        os.replace(pending, OUT/'status.json')

    def burns(self, indices):
        self.burn_number += 1
        label = f'ccm_pilot_20260913_a01_b{self.burn_number}'
        start(indices, label, OUT/f'{label}.log', 29740+self.burn_number)

    def keep_burns(self, indices):
        gpus = snapshot(indices)
        if all(not g['pids'] for g in gpus):
            self.record(event='idle_subset_restore_burn', indices=indices)
            self.burns(indices)
        else:
            inspect(indices, active=True)

    def free_after_wait(self, indices):
        time.sleep(30)
        self.record(event='free_check', gpus=require_free(indices))

    def run(self, name, command, indices=(), spare=()):
        self.stage = name
        self.record(event='launch', argv=command, indices=list(indices))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, indices)),
                   CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='offline', PYTHONUNBUFFERED='1',
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTHONPATH=str(PROJECT))
        # No inherited torchrun rendezvous configuration for the next child.
        for key in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
            env.pop(key, None)
        if indices:
            require_free(list(indices))
        with (OUT/f'{name}.log').open('x') as log:
            self.child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
            child_pid = self.child.pid
            while True:
                try:
                    code = self.child.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    # Read-only workload observation. Never signal a training child.
                    if spare:
                        self.keep_burns(list(spare))
                    self.record(event='running', pid=child_pid, gpus=snapshot(ALL),
                                task_log_tail=tail(OUT/f'{name}.log'),
                                train_tail=tail(OUT/'common/train.jsonl', 2048))
            self.child = None
        self.record(event='exit', exit_code=code, task_log_tail=tail(OUT/f'{name}.log'))
        check(code == 0, f'{name} failed with exit {code}; no automatic retry')

    def validate(self, stage):
        self.run('validate_'+stage, [PYTHON, 'scripts/validate_pilot_handoff.py', stage,
                 '--data', str(DATA), '--prep-reports', str(PREP), '--workflow', str(OUT),
                 '--assets', str(ASSETS)])

    def recover_idle(self):
        # Called only after a failure. Do not claim failed training succeeded.
        # If a child is still running, leave its GPUs alone, including cleanup.
        if self.child is not None and self.child.poll() is None:
            self.record(event='recovery_refused_live_child', pid=self.child.pid)
            return
        gpus = snapshot(ALL)
        if all(not g['pids'] for g in gpus):
            self.burns(ALL)
        elif not gpus[0]['pids'] and all(g['pids'] for g in gpus[1:]):
            # Only our verified original seven-GPU burn can be regrouped.
            inspect(SPARE)
            stop(SPARE)
            self.free_after_wait(ALL)
            self.burns(ALL)
        else:
            self.record(event='failure_workloads_preserved', gpus=gpus)

    def execute(self, prep_log):
        preflight()
        check(Path.cwd() == PROJECT and os.environ.get('CONDA_DEFAULT_ENV') == 'train_env',
              'Activate train_env in the project directory')
        check(Path(PYTHON).is_file(), 'Missing train_env')
        check(not OUT.exists() and not OUT.is_symlink(), 'Workflow output must be fresh')
        OUT.mkdir()
        self.owns_output = True
        self.run('preflight', [PYTHON, '-c',
            "from ccm.cli import code_hash; import transformers, torch; "
            f"assert code_hash() == '{CORE_SHA}'; "
            "assert transformers.__version__ == '5.9.0'; "
            "print('PILOT_CORE_AND_ENV_PREFLIGHT_PASS', torch.__version__, flush=True)"])
        self.stage = 'waiting_for_preparation'
        while not prep_finished(prep_log):
            check(prep_alive(), 'Preparation ended without its verified terminal marker')
            self.keep_burns(ALL)
            self.record(event='waiting', preparation_tail=tail(PREP/'prepare.log'), gpus=snapshot(ALL))
            time.sleep(300)
        self.validate('data')
        stop(ALL)
        self.free_after_wait(ALL)
        # The tested runtime uses torchrun, not an Accelerate configuration.
        self.free_after_wait(ALL)
        self.run('train_common', train_command(), ALL)
        self.free_after_wait(ALL)
        self.validate('common')
        self.burns(ALL)
        self.stage = 'common_verified_burns_restored'
        self.record(event='common_ready', checkpoint=str(OUT/'common/checkpoint-15259'))
        # Approved Step 3: current compiler is single-GPU. Keep 1-7 busy.
        stop(ALL)
        self.free_after_wait(ALL)
        self.burns(SPARE)
        self.run('compile_tables', compile_command(), [0], SPARE)
        self.free_after_wait([0])
        self.validate('tables')
        stop(SPARE)
        self.free_after_wait(ALL)
        self.burns(ALL)
        self.stage = 'complete'
        self.record(event='verified_steps_1_2_3_and_burns_active', success=True)
        (OUT/'complete.json').write_text((OUT/'status.json').read_text())
        # Stay alive as an idle-burn observer until the next authorized job.
        # It MUST be disarmed by STOP_IDLE_WATCH before reclaiming these burns.
        while not (OUT/'STOP_IDLE_WATCH').exists():
            time.sleep(300)
            if (OUT/'STOP_IDLE_WATCH').exists():
                break
            self.keep_burns(ALL)
            self.record(event='complete_burn_health_check', success=True, gpus=snapshot(ALL))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prep-log', type=Path, required=True)
    a = p.parse_args()
    w = Workflow()
    try:
        w.execute(a.prep_log)
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


if __name__ == '__main__':
    main()
