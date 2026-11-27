"""Dev-only, fail-closed controller download queue for authorized OHMV data.

Each batch is pre-rendered/reviewed. Submit the next #d only after the previous
batch has all item OK receipts AND its terminal OK. An optional, single
read-only node inspection reconciles ENDED-without-terminal using actual
hashes. Explicit failures always stop writes. Never retry a push automatically.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

if __package__:
    from . import a2_download_reconcile as reconcile
else:
    import a2_download_reconcile as reconcile


class MissingTerminal(RuntimeError):
    pass

def read(p): return json.loads(Path(p).read_text())


def save(p, value):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def receipts(raw, commit, count):
    lines=[x.split(' | ',5) for x in raw.splitlines()]
    # The controller can label later asynchronous events with the latest Git
    # revision, not the revision that started the download. Bind STARTED once
    # by commit, then use the immutable download task ID for all item/terminal
    # receipts. Never merge unrelated jobs that happen to share a commit.
    starts=[x for x in lines if len(x)==6 and x[1] and commit.startswith(x[1])
            and x[2]=='download' and x[4]=='STARTED']
    if not starts:return False
    ids={m.group(1) for x in starts if (m:=re.search(r'\(id=([^\)]+)\)',x[5]))}
    if len(ids)!=1 or len(starts)!=1:raise RuntimeError('Ambiguous download startup identity')
    mode='#d '+next(iter(ids))
    matched=[x for x in lines if len(x)==6 and x[2]=='download' and x[3]==mode]
    for x in matched:
        if x[4] in ('FAILED','ERROR','PARTIAL') or x[4].startswith('FAILED('):
            raise RuntimeError('Controller failure for '+commit[:8]+': '+x[4])
    terminal=[x for x in matched if x[4]=='OK' and x[5]==f'download: {count} item(s)']
    items=[x for x in matched if x[4]=='ITEM' and x[5].startswith('OK [')]
    # An unexpectedly repeated download ID or duplicate destination is not success.
    if not terminal:
        if any(x[4]=='ENDED' for x in matched):
            raise MissingTerminal('Download session ended without verified terminal success; reconcile first')
        return False
    if len(terminal)!=1 or len(items)!=count or len({x[5] for x in items})!=count:
        raise RuntimeError('Terminal/item coverage differs; reconcile first')
    return True


def run(config_path):
    config=read(config_path);state=Path(config['state']);state.mkdir(parents=True,exist_ok=True)
    worktree=Path(config['worktree']);dev=Path(config['dev_root'])
    env={**os.environ,'GIT_SSH_COMMAND':config['git_ssh_command']}
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=worktree,env=env,text=True,stderr=subprocess.STDOUT,timeout=90).strip()
    with (state/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=state/'status.json'
        if journal.exists(): raise ValueError('Existing journal: reconcile instead of restarting')
        status=dict(pid=os.getpid(),status='waiting',batch=0,commit=config['first_batch_commit'],
                    expected_head=config['initial_head'],history=[],created_utc=datetime.now(timezone.utc).isoformat())
        save(journal,status)
        def submit(command, message):
            if (state/'STOP').exists(): raise RuntimeError('STOP before command preparation')
            if git('status','--porcelain'): raise RuntimeError('Execution worktree dirty')
            if git('rev-parse','HEAD')!=status['expected_head']: raise RuntimeError('Local execution HEAD changed')
            if git('ls-remote','origin','refs/heads/main').split()[0]!=status['expected_head']:
                raise RuntimeError('Remote HEAD changed; another writer must reconcile')
            for name,h in config['source_files'].items():
                if digest(worktree/name)!=h: raise RuntimeError('Execution source changed: '+name)
            files=git('ls-files','-z').split('\0')
            size=sum((worktree/n).stat().st_size for n in files if n and n!='commands.sh')+len(command.encode())
            if size>=26214400: raise RuntimeError('Source size cap exceeded')
            (worktree/'commands.sh').write_text(command)
            git('add','commands.sh');git('diff','--cached','--check');git('commit','-m',message)
            commit=git('rev-parse','HEAD')
            status.update(status='push_in_flight',pending_push=commit);save(journal,status)
            if (state/'STOP').exists(): raise RuntimeError('STOP before push; committed intent not submitted')
            git('push','origin','HEAD:main')
            if git('ls-remote','origin','refs/heads/main').split()[0]!=commit: raise RuntimeError('Pushed HEAD unverified')
            status.update(status='waiting',expected_head=commit);save(journal,status)
            return commit

        def dropbox(*args):
            argv=['python3',str(dev/'scripts/dropbox_access.py'),'--credentials',config['credentials'],
                  '--folders',config['folders'],'--label','OHMV',*args]
            result=subprocess.run(argv,cwd=dev,capture_output=True,text=True,timeout=60)
            if result.returncode: raise RuntimeError('Dropbox retrieval failed; remote writes stopped')
            return result.stdout

        try:
            while not (state/'STOP').exists():
                stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                dest=state/(stamp+'_controller.log')
                argv=['python3',str(dev/'scripts/dropbox_access.py'),'--credentials',config['credentials'],
                    '--folders',config['folders'],'--label','OHMV','download','--path','/_RUN_STATUS_.log','--output',str(dest)]
                result=subprocess.run(argv,cwd=dev,capture_output=True,text=True,timeout=60)
                if result.returncode: raise RuntimeError('Dropbox retrieval failed; remote writes stopped')
                raw=dest.read_text()
                # This new node has no historical failures. Preserve and stop on
                # any reported infrastructure/project failure, even another commit.
                for line in raw.splitlines():
                    fields=line.split(' | ',5)
                    if len(fields)==6 and (fields[4] in ('ERROR','PARTIAL') or fields[4].startswith('FAILED')):
                        raise RuntimeError('Controller reports failure; inspect '+str(dest))
                batch=config['batches'][status['batch']]
                status.update(last_read_utc=datetime.now(timezone.utc).isoformat(),receipt=str(dest),receipt_sha256=digest(dest))
                verified=False
                if status.get('reconciliation'):
                    rec=status['reconciliation'];listing=dropbox('list')
                    entries=[json.loads(s) for s in listing.splitlines() if s.strip()]
                    matches=[x for x in entries if x.get('path','').endswith('-'+rec['payload']['job']+'.log')]
                    if len(matches)>1: raise RuntimeError('Ambiguous reconciliation log')
                    if matches:
                        if matches[0].get('bytes',0)>100000: raise RuntimeError('Unexpected reconciliation log size')
                        proof=state/(stamp+'_reconciliation.log')
                        dropbox('download','--path',matches[0]['path'],'--output',str(proof))
                        reconcile.verify(proof.read_text(),rec['payload'])
                        status['reconciliation_proof']=dict(path=str(proof),sha256=digest(proof))
                        verified=True
                    elif time.time()-rec['submitted_epoch']>1800:
                        raise RuntimeError('Reconciliation receipt timeout; no resubmission')
                else:
                    try: verified=receipts(raw,status['commit'],batch['items'])
                    except MissingTerminal:
                        if not config.get('reconcile_ended',False): raise
                        command=Path(batch['path'])
                        if digest(command)!=batch['sha256']: raise RuntimeError('Prepared command changed')
                        job='th2-ohmv-data-reconcile-'+status['commit'][:12]
                        p=reconcile.payload(dev,command.read_text(),job,status['commit'])
                        status['reconciliation']=dict(payload=p,submitted_epoch=time.time())
                        save(journal,status)
                        query_commit=submit(reconcile.command_for(p),'Read-only actual hash reconciliation for '+status['commit'][:12])
                        status['reconciliation']['query_commit']=query_commit;save(journal,status)
                if not verified:
                    save(journal,status)
                    for _ in range(config.get('poll_seconds',60)):
                        if (state/'STOP').exists():break
                        time.sleep(1)
                    continue
                status['history'].append(dict(batch=status['batch'],commit=status['commit'],receipt=str(dest),sha256=digest(dest),
                                              reconciliation=status.pop('reconciliation_proof',None)))
                status.pop('reconciliation',None)
                next_batch=status['batch']+1
                if next_batch==len(config['batches']):
                    status['status']='downloads_verified';save(journal,status);return
                if (state/'STOP').exists():break
                item=config['batches'][next_batch];command=Path(item['path'])
                if digest(command)!=item['sha256']:raise RuntimeError('Prepared download command changed')
                commit=submit(command.read_text(),f"Download verified-plan OHMV data batch {next_batch+1+config.get('batch_offset',0):02d} of 26")
                status.update(status='waiting',batch=next_batch,commit=commit);save(journal,status)
            status['status']='stopped';save(journal,status)
        except Exception as e:
            status.update(status='blocked_no_remote_writes',error=str(e),utc=datetime.now(timezone.utc).isoformat())
            save(journal,status)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    run(parser.parse_args().config)
