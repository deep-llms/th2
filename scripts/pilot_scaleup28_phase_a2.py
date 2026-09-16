"""Recovery of the externally killed Phase-A queue (2026-09-15 21:44:55 UTC).

User authorization 2026-09-16: the node is ours; another party used it by
mistake. This queue (1) stops ALL current GPU compute jobs via identity-pinned
pidfd signals after an explicit allowlist check, (2) removes the interrupted
seed-29 Shallow outputs, the unusable partial extension data root, and any
cache-*/tmp-* files under the old queue root, then (3) reruns preparation into
a fresh data root and continues: seed-29 Shallow follow-up -> smokes -> 10B
28L common training. The completed, validated seed-17 section of the old
queue is verified and carried forward, never rerun or deleted. Runs in
persistent tmux under system Python; research children use train_env.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

import pilot_overnight as runtime
from gpu_status import snapshot, require_free
from pilot_gpu_ops import BURN, check, identity, inspect, preflight, start, stop
from pilot_scaleup28_phase_a import PhaseA, common_command, verify_common_command
from scaleup28_config import (ALL, CORE, DATA28, DATA28_PARTIAL, OUT, OUT_A1, PROJECT,
                              PYTHON, SESSION, SMOKES)

runtime.OUT = OUT

AUTHORIZATION = ('User authorization 2026-09-16: node ownership confirmed; stop all '
                 'current GPU jobs left by mistaken external use and restart the '
                 'interrupted seed-29 work freshly.')
CLEANUP_TARGETS = (OUT_A1/'shallow12_seed29', OUT_A1/'shallow29_train.log', DATA28_PARTIAL)


def foreign_process_allowed(cmdline):
    """Only python compute workloads may be signaled, never arbitrary processes."""
    if not cmdline or not any(cmdline):
        return False
    joined = b' '.join(cmdline)
    if b'python' not in cmdline[0]:
        return False
    return (b'--multiprocessing-fork' in joined or b'deepeyes' in joined
            or BURN.encode() in joined or b'torch.distributed' in joined
            or b'torchrun' in joined)


def pinned_signal(pid, expected, sig):
    """Signal a pid only while its (ppid, starttime, cmdline) identity holds."""
    try:
        fd = os.pidfd_open(pid)
    except (ProcessLookupError, FileNotFoundError, OSError):
        return False
    try:
        try:
            current = identity(pid)
        except (FileNotFoundError, ProcessLookupError):
            return False
        check(current == expected, f'Process {pid} identity changed; aborting clear')
        try:
            signal.pidfd_send_signal(fd, sig)
        except ProcessLookupError:
            return False
        return True
    finally:
        os.close(fd)


class PhaseA2(PhaseA):
    def burns(self, indices):
        self.burn_number += 1
        label = f'ccm_scaleup28_phase_a2_20260916_a01_b{self.burn_number}'
        start(indices, label, OUT/f'{label}.log', 30800+self.burn_number)

    def disarm(self):
        raise RuntimeError('a2 has no previous observer to disarm; use authorized_clear')

    def no_live_own_sessions(self):
        result = subprocess.run(['tmux', 'ls', '-F', '#{session_name} #{?pane_dead,dead,alive}'],
                                capture_output=True, text=True)
        for line in (result.stdout or '').splitlines():
            name, _, state = line.partition(' ')
            if name.startswith('ccm_') and state.strip() == 'alive' and name != SESSION:
                check(False, f'Live ccm session {name}; refuse to clear GPUs under it')

    def authorized_clear(self):
        """Stop every current GPU compute job; single pass, verify, abort on failure."""
        preflight()
        self.no_live_own_sessions()
        gpus = snapshot(ALL)
        pids = sorted({pid for g in gpus for pid in g['pids']})
        if not pids:
            self.record(event='authorized_clear_not_needed')
            self.reclaimed = True
            return
        workers = {}
        for pid in pids:
            try:
                ident = identity(pid)
            except (FileNotFoundError, ProcessLookupError):
                continue
            check(foreign_process_allowed(ident[2]),
                  f'Unexpected GPU process {pid}; refuse to signal it')
            workers[pid] = ident
        parents = {}
        for pid, (ppid, _, _) in workers.items():
            if ppid > 1 and ppid not in workers and ppid not in parents:
                try:
                    ident = identity(ppid)
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if foreign_process_allowed(ident[2]):
                    parents[ppid] = ident
        self.record(event='authorized_foreign_clear', authorization=AUTHORIZATION,
                    workers={str(p): i[2][0].decode(errors='replace') for p, i in workers.items()},
                    parents={str(p): i[2][0].decode(errors='replace') for p, i in parents.items()},
                    gpus=gpus)
        for group in (parents, workers):
            for pid, ident in group.items():
                if pinned_signal(pid, ident, signal.SIGTERM):
                    self.record(event='sigterm_sent', pid=pid)
        deadline = time.monotonic()+180
        while time.monotonic() < deadline:
            if all(not g['pids'] for g in snapshot(ALL)):
                break
            time.sleep(10)
        survivors = sorted({pid for g in snapshot(ALL) for pid in g['pids']})
        for pid in survivors:
            expected = workers.get(pid) or parents.get(pid)
            check(expected is not None, f'New GPU process {pid} appeared during clear; aborting')
            if pinned_signal(pid, expected, signal.SIGKILL):
                self.record(event='sigkill_sent', pid=pid)
        time.sleep(20)
        remaining = sorted({pid for g in snapshot(ALL) for pid in g['pids']})
        check(not remaining, f'GPU processes remain after authorized clear: {remaining}')
        self.free_after_wait(ALL)
        self.free_after_wait(ALL)
        self.reclaimed = True
        self.record(event='authorized_clear_complete', gpus=require_free(ALL))

    def cleanup(self):
        """Remove ONLY the interrupted seed-29 outputs and the partial data root."""
        target = OUT_A1/'shallow12_seed29'
        if target.exists():
            check(target.is_dir() and not target.is_symlink(), 'Unexpected seed-29 target type')
            check(not (target/'complete.json').exists(), 'Refuse to delete a completed section')
            meta = json.loads((target/'train/shallow/run.json').read_text())
            check(meta['study'] == 'pilot12-shallow-followup' and meta['seed'] == 29
                  and meta['arm'] == 'shallow', 'Seed-29 target contract mismatch; refuse delete')
            shutil.rmtree(target)
            self.record(event='removed_interrupted_seed29_outputs', path=str(target))
        log = OUT_A1/'shallow29_train.log'
        if log.is_file() and not log.is_symlink():
            # Failure evidence was pulled and archived on the dev machine first.
            log.unlink()
            self.record(event='removed_interrupted_seed29_log', path=str(log))
        if DATA28_PARTIAL.exists():
            check(DATA28_PARTIAL.is_dir() and not DATA28_PARTIAL.is_symlink()
                  and not (DATA28_PARTIAL/'manifest.json').exists()
                  and not (DATA28_PARTIAL/'complete.json').exists()
                  and str(DATA28_PARTIAL) == '/mnt/local/_data/deep-llms_th2/ccm/scaleup28_v3_20260915_a01',
                  'Partial data root looks complete/unexpected; refuse delete')
            shutil.rmtree(DATA28_PARTIAL)
            self.record(event='removed_partial_extension_root', path=str(DATA28_PARTIAL))
        removed = []
        for pattern in ('cache-*', 'tmp-*'):
            for stale in OUT_A1.rglob(pattern):
                if stale.is_file() and not stale.is_symlink():
                    stale.unlink()
                    removed.append(str(stale))
        self.record(event='removed_cache_tmp_files', files=removed, count=len(removed))

    def execute(self):
        preflight()
        check(Path.cwd() == PROJECT and os.environ.get('CONDA_DEFAULT_ENV') == 'train_env',
              'Wrong project/environment')
        check(not OUT.exists() and not OUT.is_symlink(), 'a2 output must be fresh')
        check(not DATA28.exists() and not DATA28.is_symlink(), 'a2 data root must be fresh')
        check(OUT_A1.is_dir(), 'Prior queue root missing; nothing to recover from')
        OUT.mkdir()
        self.owns_output = True
        self.run('preflight', [PYTHON, '-c',
            "from ccm.cli import code_hash; import transformers, torch; "
            f"assert code_hash() == '{CORE}'; "
            "assert transformers.__version__ == '5.9.0'; "
            "print('SCALEUP28_A2_CORE_AND_ENV_PREFLIGHT_PASS', torch.__version__, flush=True)"])
        self.validate('inputs')
        self.authorized_clear()
        self.cleanup()
        self.start_preparation()
        self.shallow_followup(29)
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
        self.record(event='scaleup28_phase_a2_verified_and_burns_active', success=True,
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
    w = PhaseA2()
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
