"""One owner: verify existing common -> compile -> Stage1 -> Stage2 -> burns."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import pilot_overnight as runtime
from pilot_gpu_ops import check, identity, inspect, preflight, start, stop
from gpu_status import snapshot
from seed29_replication_config import (SEED, COMMON, OUT, PREVIOUS, OLD_SESSION, SESSION,
    DATA, PREP, PROJECT, PYTHON, ALL, SPARE, STAGE1_ARMS, DECISION, stage2_arms)

runtime.OUT = OUT  # This process only; existing A3 observer has its own module state.


def compile_command():
    return [PYTHON, '-m', 'ccm', 'compile', '--data', str(DATA/'corpus'),
            '--vocabulary', str(DATA/'vocabulary.npz'), '--checkpoint',
            str(COMMON/'common/checkpoint-15259'), '--output', str(OUT/'tables'),
            '--device', 'cuda', '--microbatch-segments', '4', '--isolated-batch', '256']


def train_command(phase, arm):
    check(phase in ('stage1','stage2'), 'Wrong phase')
    allowed = STAGE1_ARMS if phase == 'stage1' else stage2_arms(json.loads(DECISION.read_text()))
    check(arm in allowed, 'Unscheduled training arm')
    argv = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
            '--module', 'ccm', 'train', '--phase', phase, '--arm', arm, '--seed', str(SEED),
            '--data', str(DATA/'corpus'), '--vocabulary', str(DATA/'vocabulary.npz'),
            '--coverage', str(PREP/'coverage.json'), '--checkpoint', str(COMMON/'common/checkpoint-15259'),
            '--output', str(OUT/phase/'train'/arm), '--device', 'cuda', '--activation-checkpointing',
            '--microbatch-segments', '8', '--loss-chunk', '1024', '--save-every', '1000', '--log-every', '10']
    if arm not in ('base','grad'):
        argv += ['--table', str(OUT/'tables'/arm)]
    if phase == 'stage2' and arm == 'delta':
        argv += ['--delta-decision', str(DECISION)]
    return argv


def eval_command(phase, arm):
    ck = COMMON/'common/checkpoint-15259' if phase == 'stage1' and arm == 'base' else (
        OUT/phase/'train'/arm/f'checkpoint-{977 if phase == "stage1" else 3815}')
    return [PYTHON, '-m', 'ccm', 'evaluate', '--data', str(DATA/'corpus'),
            '--vocabulary', str(DATA/'vocabulary.npz'), '--checkpoint', str(ck),
            '--output', str(OUT/phase/'eval'/arm), '--role', 'dev', '--device', 'cuda',
            '--microbatch-segments', '8', '--loss-chunk', '1024',
            '--diagnostic-table', str(OUT/'tables/contextual')]


def observer():
    if subprocess.run(['tmux','has-session','-t',OLD_SESSION],capture_output=True).returncode:
        return None
    info = subprocess.check_output(['tmux','display-message','-p','-t',OLD_SESSION,
                                    '#{pane_dead} #{pane_pid}'],text=True).split()
    if info[0] == '1':
        return None
    pid = int(info[1])
    try:
        ident = identity(pid)
    except (FileNotFoundError, ProcessLookupError):
        return None
    if not any(ident[2]):
        return None
    check(b'scripts/pilot_a3_diagnostics.py' in ident[2], 'Unexpected prior observer')
    return pid, ident


class Replication(runtime.Workflow):
    def __init__(self):
        super().__init__()
        self.reclaimed = False

    def burns(self, indices):
        self.burn_number += 1
        label = f'ccm_seed29_replication_20260914_a01_b{self.burn_number}'
        start(indices,label,OUT/f'{label}.log',30340+self.burn_number)

    def keep_burns(self, indices):
        if self.reclaimed:
            super().keep_burns(indices)
        else:
            inspect(indices,active=True)

    def recover_idle(self):
        if not self.reclaimed:
            self.record(event='failure_preserved_previous_burn_owner')
            return
        super().recover_idle()

    def validate(self, action, phase=None, arm=None, spare=()):
        argv = [PYTHON,'scripts/validate_seed29_replication.py',action]
        if phase:
            argv += ['--phase',phase]
        if arm:
            argv += ['--arm',arm]
        self.run('_'.join(x for x in ('validate',phase,action,arm) if x),argv,spare=spare)

    def disarm(self):
        done = json.loads((PREVIOUS/'complete.json').read_text())
        status = json.loads((PREVIOUS/'status.json').read_text())
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(status['time'])).total_seconds()
        check(done.get('success') is True and done.get('event') == 'a3_diagnostics_verified_and_burns_active'
              and status.get('success') is True and status['stage'] == 'complete' and -10 <= age <= 180,
              'A3 completion/heartbeat invalid; burns untouched')
        current = observer()
        check(current is not None, 'Expected A3 observer missing; inspect before takeover')
        inspect(ALL,active=True)
        flag = PREVIOUS/'STOP_IDLE_WATCH'
        check(not flag.exists() and not flag.is_symlink(),'A3 observer already disarmed')
        flag.open('x').close()
        pid, expected = current
        for _ in range(12):
            try:
                alive = identity(pid) == expected
            except (FileNotFoundError,ProcessLookupError):
                alive = False
            if not alive:
                break
            self.record(event='waiting_for_previous_observer_exit',pid=pid)
            time.sleep(30)
        check(observer() is None,'A3 observer remains live; burns untouched')
        self.reclaimed = True
        self.record(event='previous_observer_exited_without_process_signal')

    def compare(self, phase, left, right):
        root = OUT/phase
        self.run(f'{phase}_compare_{left}_{right}',[PYTHON,'-m','ccm','compare',
            '--left',str(root/'eval'/left),'--right',str(root/'eval'/right),
            '--population','overall','--cluster','doc_id','--cross-seed-coupling','independent',
            '--replicates','10000','--output',str(root/'reports'/f'{left}_vs_{right}.json')],spare=ALL)

    def phase(self, name, arms):
        for arm in arms:
            self.run(f'{name}_train_{arm}',train_command(name,arm),ALL)
            self.free_after_wait(ALL)
            self.validate('train',name,arm)
        self.burns(SPARE)
        for arm in (('base',)+arms if name == 'stage1' else arms):
            self.run(f'{name}_eval_{arm}',eval_command(name,arm),[0],SPARE)
            self.free_after_wait([0])
            self.validate('eval',name,arm,spare=SPARE)
        stop(SPARE)
        self.free_after_wait(ALL)
        self.burns(ALL)
        for right in (('isolated','shuffled') if name == 'stage1' else ('isolated','shuffled','base','grad')):
            self.compare(name,'contextual',right)
        if name == 'stage1':
            self.run('stage1_delta_decision',[PYTHON,'-m','ccm','delta-decision',
                '--delta',str(OUT/'stage1/eval/delta'),'--contextual',str(OUT/'stage1/eval/contextual'),
                '--shuffled',str(OUT/'stage1/eval/shuffled'),'--cluster','doc_id',
                '--replication-policy','per_seed','--output',str(DECISION)],spare=ALL)
        elif 'delta' in arms:
            for right in ('contextual','isolated','shuffled'):
                self.compare(name,'delta',right)
        self.validate('panel',name,spare=ALL)
        self.record(event=name+'_validated',success=True,arms=list(arms))
        with (OUT/name/'complete.json').open('x') as f:
            f.write((OUT/'status.json').read_text())

    def execute(self):
        preflight()
        check(Path.cwd() == PROJECT and os.environ.get('CONDA_DEFAULT_ENV') == 'train_env',
              'Wrong project/environment')
        check(not OUT.exists() and not OUT.is_symlink(),'Replication output must be fresh')
        OUT.mkdir()
        self.owns_output = True
        for phase in ('stage1','stage2'):
            for sub in ('train','eval','reports'):
                (OUT/phase/sub).mkdir(parents=True,exist_ok=True)
        self.validate('inputs',spare=ALL)
        self.disarm()
        stop(ALL)
        self.free_after_wait(ALL)
        self.free_after_wait(ALL)
        self.burns(SPARE)
        self.run('compile_tables',compile_command(),[0],SPARE)
        self.free_after_wait([0])
        self.validate('tables',spare=SPARE)
        stop(SPARE)
        self.free_after_wait(ALL)
        self.free_after_wait(ALL)
        self.phase('stage1',STAGE1_ARMS)
        # One owner performs the transition; no separate observer can restart burns.
        arms = stage2_arms(json.loads(DECISION.read_text()))
        stop(ALL)
        self.free_after_wait(ALL)
        self.free_after_wait(ALL)
        self.phase('stage2',arms)
        inspect(ALL,active=True)
        self.stage = 'complete'
        self.record(event='seed29_replication_verified_and_burns_active',success=True,
                    stage2_arms=list(arms),gpus=snapshot(ALL))
        with (OUT/'complete.json').open('x') as f:
            f.write((OUT/'status.json').read_text())
        while not (OUT/'STOP_IDLE_WATCH').exists():
            time.sleep(60)
            if (OUT/'STOP_IDLE_WATCH').exists():
                break
            self.keep_burns(ALL)
            self.record(event='complete_burn_health_check',success=True,gpus=snapshot(ALL))


if __name__ == '__main__':
    w = Replication()
    try:
        w.execute()
    except Exception as error:
        traceback.print_exc()
        if w.owns_output:
            w.stage = 'failed'
            w.record(event='failure',error=str(error),success=False)
            try:
                w.recover_idle()
            except Exception:
                traceback.print_exc()
        raise SystemExit(1)
