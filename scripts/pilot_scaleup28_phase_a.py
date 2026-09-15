"""One owner for plan v3 §23.A: extension preparation (background CPU) plus the
two 12L Shallow follow-ups, then the decided 28L smoke/stability/10B common queue.

Runs in persistent tmux under system Python (pidfd support); research children
use the activated train_env. Never retries training, never deletes output, and
touches GPU processes only through the verified-burn-worker procedure. The 28L
panel (compile/Stage-1/Stage-2), 12L locked D_val evaluations, and any later
seed are NOT launched by this queue.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import pilot_overnight as runtime
from pilot_gpu_ops import check, identity, inspect, preflight, start, stop
from gpu_status import snapshot
from scaleup28_config import (ALL, ASSETS, COMMON_ROOT, COMMON_SAVE_EVERY, CORE, DATA, DATA28,
    FOLLOWUP, CONTEXT_EVAL, OLD_EVENT, OLD_SCRIPT, OLD_SESSION, OUT, PREP, PREVIOUS, PROJECT,
    PYTHON, RAW, REVISION, SEED28, SESSION, SHALLOW_SEEDS, SMOKES, SPARE, STUDY, TABLES,
    shallow_root)

runtime.OUT = OUT  # This process only; the previous observer keeps its own module state.


def prepare_command():
    return [PYTHON, '-m', 'ccm', 'prepare-scaleup', '--historical-data', str(DATA/'corpus'),
            '--vocabulary', str(DATA/'vocabulary.npz'), '--raw-dir', str(RAW),
            '--source-manifest', 'resources/culturax_raw_manifest.tsv',
            '--dataset-revision', REVISION, '--tokenizer-path', str(ASSETS),
            '--tokenizer-manifest', 'resources/qwen3_base_assets.json', '--output', str(DATA28)]


def shallow_train_command(seed):
    root = shallow_root(seed)
    return [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
            '-m', 'ccm', 'train', '--study', FOLLOWUP, '--phase', 'stage2', '--arm', 'shallow',
            '--seed', str(seed), '--data', str(DATA/'corpus'), '--vocabulary', str(DATA/'vocabulary.npz'),
            '--coverage', str(PREP/'coverage.json'), '--checkpoint', str(COMMON_ROOT[seed]/'common/checkpoint-15259'),
            '--table', str(TABLES[seed]/'shallow'), '--output', str(root/'train/shallow'),
            '--device', 'cuda', '--activation-checkpointing', '--microbatch-segments', '8',
            '--loss-chunk', '1024', '--save-every', '1000', '--log-every', '10']


def shallow_eval_command(seed):
    root = shallow_root(seed)
    return [PYTHON, '-m', 'ccm', 'evaluate', '--data', str(DATA/'corpus'),
            '--vocabulary', str(DATA/'vocabulary.npz'),
            '--checkpoint', str(root/'train/shallow/checkpoint-3815'),
            '--output', str(root/'eval/shallow'), '--role', 'dev', '--device', 'cuda',
            '--microbatch-segments', '8', '--loss-chunk', '1024',
            '--diagnostic-table', str(TABLES[seed]/'contextual')]


def shallow_compare_command(seed):
    root = shallow_root(seed)
    return [PYTHON, '-m', 'ccm', 'compare', '--left', str(CONTEXT_EVAL[seed]),
            '--right', str(root/'eval/shallow'), '--population', 'overall',
            '--cluster', 'doc_id', '--cross-seed-coupling', 'independent',
            '--replicates', '10000', '--output', str(root/'reports/contextual_vs_shallow.json')]


def common_command(name, stability=None):
    # Must stay byte-equal to the decided `ccm scaleup-jobs --queue common`
    # manifest after {python}/{run_dir} substitution; tests enforce this.
    argv = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
            '-m', 'ccm', 'train', '--study', STUDY, '--data', str(DATA28), '--phase', 'common',
            '--arm', 'base', '--seed', str(SEED28), '--output', str(OUT/name), '--device', 'cuda',
            '--microbatch-segments', '8', '--loss-chunk', '1024', '--activation-checkpointing',
            '--save-every', str(stability if stability else COMMON_SAVE_EVERY),
            '--model-config', str(ASSETS), '--common-lr', '0.0003']
    if stability:
        argv += ['--stability-steps', str(stability)]
    else:
        argv += ['--stability-report', str(OUT/'common-stability/stability.json')]
    return argv


def verify_common_command():
    return [PYTHON, '-m', 'ccm', 'validate-run', '--data', str(DATA28),
            '--run', str(OUT/'common-base'), '--phase', 'common', '--arm', 'base',
            '--seed', str(SEED28), '--output', str(OUT/'verified-common-base.json')]


def previous_state_ok(done, status, age_seconds):
    return (done.get('success') is True and done.get('event') == OLD_EVENT
            and status.get('success') is True and status.get('stage') == 'complete'
            and -10 <= age_seconds <= 180)


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
    check(OLD_SCRIPT in ident[2], 'Unexpected prior observer')
    return pid, ident


class PhaseA(runtime.Workflow):
    def __init__(self):
        super().__init__()
        self.reclaimed = False
        self.prep = None

    def record(self, **fields):
        if self.prep is not None and 'prepare_running' not in fields:
            fields['prepare_running'] = self.prep.poll() is None
        super().record(**fields)

    def burns(self, indices):
        self.burn_number += 1
        label = f'ccm_scaleup28_phase_a_20260915_a01_b{self.burn_number}'
        start(indices, label, OUT/f'{label}.log', 30560+self.burn_number)

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

    def stop_own_preparation(self):
        # Our own CPU child only; a partial extension has no manifest and is
        # preserved on disk for inspection, never deleted or resumed.
        if self.prep is None or self.prep.poll() is not None:
            return
        self.record(event='terminating_own_preparation_child', pid=self.prep.pid)
        self.prep.send_signal(signal.SIGTERM)
        try:
            self.prep.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.prep.kill()
            self.prep.wait(timeout=60)

    def validate(self, action, seed=None, name=None, spare=()):
        argv = [PYTHON, 'scripts/validate_scaleup28_phase_a.py', action]
        if seed is not None:
            argv += ['--seed', str(seed)]
        if name is not None:
            argv += ['--name', name]
        label = '_'.join(str(x) for x in ('validate', action, seed, name) if x is not None)
        self.run(label.replace('-', '_'), argv, spare=spare)

    def disarm(self):
        done = json.loads((PREVIOUS/'complete.json').read_text())
        status = json.loads((PREVIOUS/'status.json').read_text())
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(status['time'])).total_seconds()
        check(previous_state_ok(done, status, age), 'Seed29 completion/heartbeat invalid; burns untouched')
        current = observer()
        check(current is not None, 'Expected seed29 observer missing; inspect before takeover')
        inspect(ALL, active=True)
        flag = PREVIOUS/'STOP_IDLE_WATCH'
        check(not flag.exists() and not flag.is_symlink(), 'Seed29 observer already disarmed')
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
        check(observer() is None, 'Seed29 observer remains live; burns untouched')
        self.reclaimed = True
        self.record(event='previous_observer_exited_without_process_signal')

    def start_preparation(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='offline', PYTHONUNBUFFERED='1',
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   PYTHONPATH=str(PROJECT))
        for key in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
            env.pop(key, None)
        log = (OUT/'prepare_scaleup.log').open('x')
        self.prep = subprocess.Popen(prepare_command(), cwd=PROJECT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
        self.record(event='preparation_started_background_cpu', pid=self.prep.pid,
                    argv=prepare_command())

    def wait_for_preparation(self):
        self.stage = 'waiting_for_preparation'
        while self.prep.poll() is None:
            self.keep_burns(ALL)
            self.record(event='waiting_for_preparation',
                        preparation_tail=runtime.tail(OUT/'prepare_scaleup.log'), gpus=snapshot(ALL))
            time.sleep(120)
        code = self.prep.returncode
        self.record(event='preparation_exit', exit_code=code,
                    preparation_tail=runtime.tail(OUT/'prepare_scaleup.log'))
        check(code == 0, f'Extension preparation failed with exit {code}; no automatic retry')

    def shallow_followup(self, seed):
        root = shallow_root(seed)
        for sub in ('train', 'eval', 'reports'):
            (root/sub).mkdir(parents=True, exist_ok=True)
        self.run(f'shallow{seed}_train', shallow_train_command(seed), ALL)
        self.free_after_wait(ALL)
        self.validate('shallow-train', seed=seed)
        self.burns(SPARE)
        self.run(f'shallow{seed}_eval', shallow_eval_command(seed), [0], SPARE)
        self.free_after_wait([0])
        self.validate('shallow-eval', seed=seed, spare=SPARE)
        stop(SPARE)
        self.free_after_wait(ALL)
        self.burns(ALL)
        self.run(f'shallow{seed}_compare', shallow_compare_command(seed), spare=ALL)
        self.validate('followup', seed=seed, spare=ALL)
        self.record(event=f'shallow12_seed{seed}_validated', success=True)
        with (root/'complete.json').open('x') as f:
            f.write((OUT/'status.json').read_text())
        stop(ALL)
        self.free_after_wait(ALL)
        self.free_after_wait(ALL)

    def execute(self):
        preflight()
        check(Path.cwd() == PROJECT and os.environ.get('CONDA_DEFAULT_ENV') == 'train_env',
              'Wrong project/environment')
        check(not OUT.exists() and not OUT.is_symlink(), 'Phase-A output must be fresh')
        check(not DATA28.exists() and not DATA28.is_symlink(), 'Extension data root must be fresh')
        OUT.mkdir()
        self.owns_output = True
        self.run('preflight', [PYTHON, '-c',
            "from ccm.cli import code_hash; import transformers, torch; "
            f"assert code_hash() == '{CORE}'; "
            "assert transformers.__version__ == '5.9.0'; "
            "print('SCALEUP28_CORE_AND_ENV_PREFLIGHT_PASS', torch.__version__, flush=True)"])
        self.validate('inputs')
        self.start_preparation()
        self.disarm()
        stop(ALL)
        self.free_after_wait(ALL)
        self.free_after_wait(ALL)
        for seed in SHALLOW_SEEDS:
            self.shallow_followup(seed)
        self.burns(ALL)
        self.wait_for_preparation()
        self.run('validate_data_extension', [PYTHON, '-m', 'ccm', 'validate-data',
                 '--data', str(DATA28)], spare=ALL)
        self.validate('extension', spare=ALL)
        stop(ALL)
        self.free_after_wait(ALL)
        self.free_after_wait(ALL)
        for name, steps in SMOKES:
            self.run(name.replace('-', '_'), common_command(name, steps), ALL)
            self.free_after_wait(ALL)
            self.validate('smoke', name=name.split('-')[1])
        self.run('common_base', common_command('common-base'), ALL)
        self.free_after_wait(ALL)
        self.burns(ALL)
        self.run('verify_common_base', verify_common_command(), spare=ALL)
        self.validate('common', spare=ALL)
        inspect(ALL, active=True)
        self.stage = 'complete'
        self.record(event='scaleup28_phase_a_verified_and_burns_active', success=True,
                    theta_10b_28l=str(OUT/'common-base/checkpoint-38147'), gpus=snapshot(ALL))
        with (OUT/'complete.json').open('x') as f:
            f.write((OUT/'status.json').read_text())
        while not (OUT/'STOP_IDLE_WATCH').exists():
            time.sleep(60)
            if (OUT/'STOP_IDLE_WATCH').exists():
                break
            self.keep_burns(ALL)
            self.record(event='complete_burn_health_check', success=True, gpus=snapshot(ALL))


if __name__ == '__main__':
    w = PhaseA()
    try:
        w.execute()
    except Exception as error:
        traceback.print_exc()
        if w.owns_output:
            w.stage = 'failed'
            w.record(event='failure', error=str(error), success=False)
            try:
                w.stop_own_preparation()
            except Exception:
                traceback.print_exc()
            try:
                w.recover_idle()
            except Exception:
                traceback.print_exc()
        raise SystemExit(1)
