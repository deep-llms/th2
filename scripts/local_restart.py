"""Gated four-A100 teacher restart, matched students, and verified filesystem backups.

Run from an immutable source snapshot. All model jobs remain foreground children;
this controller can itself be hosted in tmux. No remote runner or process reclaim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.part')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


class Backup:
    """Copy an open checkpoint inode, then verify destination before publication.

    Trainers atomically replace latest.pt. Opening it once prevents a concurrent
    replacement from producing mixed bytes. A second filesystem is required by
    the production controller; no network credentials or upload tool is used.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.records = {}
        self.lock = threading.Lock()
        self.error = None

    def copy(self, source, relative):
        source = Path(source)
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Backup destination must be relative and confined')
        with self.lock, source.open('rb') as src:
            st = os.fstat(src.fileno())
            identity = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]
            old = self.records.get(str(relative))
            if old and old['source_identity'] == identity:
                return old
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.part')
            sha = hashlib.sha256()
            with partial.open('wb') as dest:
                while block := src.read(8 * 1024 * 1024):
                    sha.update(block); dest.write(block)
                dest.flush(); os.fsync(dest.fileno())
            after = os.fstat(src.fileno())
            if identity != [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns]:
                raise ValueError(f'Source changed during backup: {source}')
            with partial.open('rb') as check:
                verified = hashlib.file_digest(check, 'sha256').hexdigest()
            if verified != sha.hexdigest() or partial.stat().st_size != st.st_size:
                raise ValueError(f'Backup verification failed: {target}')
            os.replace(partial, target)
            record = {'source':str(source.resolve()), 'source_identity':identity,
                      'bytes':st.st_size, 'sha256':verified, 'verified_utc':stamp()}
            self.records[str(relative)] = record
            atomic_json(self.root/'manifest.json', {'status':'ok', 'files':self.records})
            print(f'BACKUP_VERIFIED {relative} bytes={st.st_size}', flush=True)
            return record


def require_separate_filesystem(root, backup_root):
    backup_root.parent.mkdir(parents=True, exist_ok=True)
    if backup_root.parent.stat().st_dev == root.stat().st_dev:
        raise ValueError('Backups must live on a different filesystem')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--backup-root', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    from pcc.joint_config import load_config
    from pcc.joint_training import code_identity
    config = load_config(root/'config.json')
    if config['experiment'] != 'joint-local-v3':
        raise ValueError('This controller requires the fixed local restart plan')
    if json.loads((root/'cpu-ready.json').read_text()) != {'status':'ok','code':code_identity()}:
        raise ValueError('CPU readiness/source mismatch')
    if not (root/'inputs/complete.json').is_file():
        raise ValueError('Prepared inputs must be complete before launch')
    if (root/'pipeline.json').exists():
        raise ValueError('Refusing to replay an existing pipeline')
    require_separate_filesystem(root, args.backup_root)
    backup = Backup(args.backup_root)
    state = {'status':'running','started_utc':stamp(),'stage':'backing_up_inputs',
             'root':str(root),'backup_root':str(args.backup_root.resolve()),'pid':os.getpid()}
    atomic_json(root/'pipeline.json',state)
    stop = threading.Event()

    def checkpoint_paths():
        for pattern in ('teacher-seed-*/runs/seed-*-Deep/latest.pt', 'students/seed-*/latest.pt'):
            yield from root.glob(pattern)

    def checkpoint_backup():
        try:
            while not stop.is_set():
                for path in checkpoint_paths():
                    backup.copy(path, path.relative_to(root))
                stop.wait(15)
        except BaseException as error:
            backup.error = str(error)
            atomic_json(root/'backup-failure.json', {'error':str(error),'time':stamp()})
            print(f'BACKUP_FAILURE: {error}; no next stage will start', flush=True)

    def run(name, argv):
        if backup.error:
            raise RuntimeError('Backup failed: '+backup.error)
        state.update(stage=name, stage_started_utc=stamp(), argv=argv)
        atomic_json(root/'pipeline.json', state)
        print(f'STAGE_START {name}', flush=True)
        with (root/(name+'.log')).open('x') as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
        if backup.error:
            raise RuntimeError('Backup failed: '+backup.error)
        backup.copy(root/(name+'.log'), name+'.log')
        print(f'STAGE_COMPLETE {name}', flush=True)

    def archive(directory):
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.name != 'latest.pt' and not path.name.endswith('.part'):
                backup.copy(path, path.relative_to(root))

    def complete(decision):
        stop.set()
        if thread is not None:
            thread.join()
        if backup.error:
            raise RuntimeError('Backup failed: '+backup.error)
        finished = {**state, 'status':'ok', 'decision':decision, 'completed_utc':stamp()}
        # A failed final backup must not leave a successful local completion.
        atomic_json(root/'completion-candidate.json', finished)
        backup.copy(root/'completion-candidate.json', 'complete.json')
        atomic_json(root/'complete.json', finished)
        state.update(finished)
        atomic_json(root/'pipeline.json', state)

    thread = None
    try:
        for name in ('config.json','source.json','cpu-ready.json','PLAN.md'):
            backup.copy(root/name,name)
        for name in ('launch.json', 'launch.sh', 'regression-v2.log', 'local-tests-v3.log'):
            if (root/name).is_file():
                backup.copy(root/name, name)
        for path in (root/'source').rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                backup.copy(path,path.relative_to(root))
        for name in ('complete.json','train.npz','dev.npz'):
            backup.copy(root/'inputs'/name,Path('inputs')/name)
        assets = json.loads(Path('resources/qwen3_joint_assets.json').read_text())
        for item in assets['files']:
            record = backup.copy(Path(config['model_path'])/item['path'], Path('pretrained')/item['path'])
            if record['sha256'] != item['sha256'] or record['bytes'] != item['bytes']:
                raise ValueError('Pinned model checksum mismatch')
        thread = threading.Thread(target=checkpoint_backup, daemon=True)
        thread.start()
        python = sys.executable
        shared = ['--config',str(root/'config.json'),'--data-dir',str(root/'inputs')]
        gpu = ['--physical-gpus','0','1','2','3']
        run('capacity-Deep',[python,'-u','-m','pcc.joint','capacity',*shared,
                            '--arm','Deep','--seed-index','0',*gpu,'--resume-check',
                            '--output',str(root/'capacity-Deep')])
        cap = json.loads((root/'capacity-Deep/capacity.json').read_text())
        if not (cap['status']=='ok' and cap['world_size']==4 and cap['resume_next_update_verified']
                and cap['initial_native_equivalence'] and cap['checkpoint_roundtrip']
                and all(cap['parameters_changed'].values())):
            raise ValueError('Deep capacity/resume failed')
        for seed in (0,1):
            joint = root/f'teacher-seed-{seed}'
            joint.mkdir(exist_ok=False)
            name = f'seed-{seed}-Deep'
            jobs = {'jobs':[{'name':name,'gpus':list(range(4)),
                'argv':['{python}','-u','-m','pcc.joint','train',*shared,'--arm','Deep',
                        '--seed-index',str(seed),*gpu,'--output','{run_dir}/'+name],
                'required_outputs':[{'path':name+'/complete.json','json_equals':{
                    'status':'ok','arm':'Deep','updates':6144,'input_tokens':201326592,
                    'world_size':4,'checkpoint_verified':True}}]}]}
            atomic_json(joint/'jobs.json', jobs)
            run(f'teacher-seed-{seed}', [python,'-u','run_experiments.py','--config',str(joint/'jobs.json'),
                '--project-dir',str(Path.cwd()),'--run-dir',str(joint/'runs')])
            # Full parent weights and receipts must be durable before students start.
            archive(joint)
            base = [*shared,'--joint-root',str(joint),'--seed-index',str(seed),*gpu]
            audit_dir = root/f'audit-seed-{seed}'
            run(f'audit-seed-{seed}',[python,'-u','-m','pcc.distill','audit',*base,'--output',str(audit_dir)])
            archive(audit_dir)
            audit = json.loads((audit_dir/'complete.json').read_text())
            if not (audit['status']=='ok' and audit['ready_for_students']):
                complete(f'stop_no_feedback_benefit_seed_{seed}')
                return
            for arm in ('LM','PCC'):
                common = [*base,'--arm',arm,'--audit-dir',str(audit_dir)]
                if seed == 0:
                    capacity = root/f'capacity-{arm}'
                    run(f'capacity-{arm}',[python,'-u','-m','pcc.distill','capacity',*common,
                                         '--output',str(capacity)])
                    cap=json.loads((capacity/'capacity.json').read_text())
                    if not (cap['status']=='ok' and cap['world_size']==4 and cap['teacher_unchanged']
                            and cap['student_changed'] and cap['checkpoint_roundtrip']
                            and cap['resume_next_update_verified'] and cap['student_noop_exact']):
                        raise ValueError('Student capacity/resume failed')
                directory=root/'students'/f'seed-{seed}-{arm}'
                run(f'seed-{seed}-{arm}',[python,'-u','-m','pcc.distill','train',*common,'--output',str(directory)])
                archive(directory)
            report=root/f'report-{seed+1}-seed'
            run(f'report-{seed+1}-seed',[python,'-u','-m','pcc.distill','report',*shared,
                '--joint-root',str(joint),'--audit-root',str(root),'--runs-dir',str(root/'students'),
                '--seed-count',str(seed+1),'--output',str(report)])
            archive(report)
            result=json.loads((report/'complete.json').read_text())
            if not all(row['passed'] for row in result['seeds']):
                complete('stop_no_consistent_student_recovery')
                return
        complete('recommend_target_semantics_control')
    except BaseException as error:
        state.update(status='failed',error=str(error),failed_utc=stamp())
        atomic_json(root/'pipeline.json', state)
        raise
    finally:
        stop.set()
        if thread is not None:
            thread.join()


if __name__ == '__main__':
    main()
