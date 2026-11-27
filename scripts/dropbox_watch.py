"""Local metadata-only Dropbox watcher:120s polling,60s stability debounce.

No remote jobs, downloads, GPU operations or LLM calls while polling. Stable
changes request one wake through the existing queue watcher, then polling resumes.
With --stdout (config notify="stdout") each stable change or HOLD is instead printed
as ONE stdout line, for an agent-side line monitor (e.g. Claude Code Monitor).
Credentials and configuration remain private. API errors hold rather than retry.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import uuid


class PollError(RuntimeError):
    """Public diagnostics constructed locally, never raw helper/API text."""
    def __init__(self, reason, *, http_status=None):
        super().__init__('Dropbox metadata check failed')
        self.reason=reason
        self.http_status=http_status


def helper_failure(stderr):
    # Match the helper's entire known-safe line; never forward arbitrary stderr.
    message=stderr.strip()
    http=re.fullmatch(r'Dropbox retrieval failed: Dropbox HTTP ([1-5][0-9]{2}); remote status is unverified',message)
    if http:return PollError('dropbox_http',http_status=int(http.group(1)))
    if message=='Dropbox retrieval failed: Dropbox connection failed; remote status is unverified':
        return PollError('dropbox_connection')
    return PollError('helper_failed')


def write(path,data):
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
    finally:tmp.unlink(missing_ok=True)


def fingerprint(snapshot):
    return hashlib.sha256(json.dumps(snapshot,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def observe(state,snapshot,now,settle=60):
    """Return a stable change once. Baseline initialization never sends a wake."""
    sig=fingerprint(snapshot)
    if 'baseline' not in state:
        state.update(baseline=snapshot,baseline_hash=sig,candidate_hash=None)
        return None
    if sig==state['baseline_hash']:
        state.update(candidate_hash=None);return None
    if sig!=state.get('candidate_hash'):
        state.update(candidate_hash=sig,candidate_since=now);return None
    if now-state['candidate_since']<settle:return None
    old=state['baseline']
    changed=sorted(k for k in old.keys()|snapshot.keys() if old.get(k)!=snapshot.get(k))
    event={'revision':'dropbox-'+uuid.uuid4().hex,'snapshot_sha256':sig,
           'observed_utc':datetime.fromtimestamp(now,timezone.utc).isoformat(),
           'changed_count':len(changed),'changed_paths':changed[:30]}
    state.update(baseline=snapshot,baseline_hash=sig,candidate_hash=None)
    return event


def snapshot(config):
    helper=Path(__file__).with_name('dropbox_access.py').resolve()
    def listing(path):
        command=[sys.executable,str(helper),'--credentials',config['credentials'],
                 '--folders',config['folders'],'--label',config['label'],
                 'list','--path',path,'--identity']
        try:
            result=subprocess.run(command,cwd=config['cwd'],capture_output=True,text=True,timeout=60)
        except subprocess.TimeoutExpired:
            raise PollError('helper_timeout') from None
        if result.returncode:raise helper_failure(result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    # User-selected status file only; root metadata listing, no subtree scans.
    target=config['status_path']
    if not target.startswith('/') or '/' in target[1:]:raise ValueError('Root status file required')
    return {e['path']:e for e in listing('') if e['path']==target and e['type']=='file'}


def notify(state_dir,watcher,event):
    # Durable event first; one request only. queue watcher owns actual delivery.
    write(state_dir/(event['revision']+'.json'),event)
    status=json.loads((watcher/'status.json').read_text())
    pid=status['pid'];os.kill(pid,0)  # existence check only, never signals a process
    if status.get('status') not in ('waiting','running'):raise RuntimeError('Queue watcher unavailable')
    message=('Dropbox có thay đổi metadata đã ổn định qua hai lần kiểm tra cách nhau ít nhất1phút. '
             'Đọc AGENTS.md/docs/CURRENT_TASK.md; xem '+str(state_dir/(event['revision']+'.json'))+
             '. Retrieve/reconcile receipt theo job/commit trước writes; đây KHÔNG phải bằng chứng job hoàn tất. '
             'Không resend/relaunch. Tuân thủ quyền hiện hành trong CURRENT_TASK.md. Ghi actual delivery khi nhận; '
             'poller sẽ tiếp tục sau2phút, không tạo poller mới.')
    write(watcher/'schedule.json',{'revision':event['revision'],
                                 'next_utc':datetime.now(timezone.utc).isoformat(),'message':message})


def notify_stdout(event,current):
    # One line per stable change; metadata only, no credentials or URLs.
    entries=';'.join('%s bytes=%s modified=%s'%(p,e.get('bytes'),e.get('modified')) for p,e in sorted(current.items()))
    print('DROPBOX_STABLE_CHANGE %s observed=%s %s'%(event['revision'],event['observed_utc'],entries),flush=True)


def run(config,state_dir,stop_after=None):
    state_dir.mkdir(parents=True,exist_ok=True)
    with (state_dir/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cache=state_dir/'observation.json'
        state=json.loads(cache.read_text()) if cache.exists() else {}
        if (state_dir/'HOLD').exists():raise RuntimeError('Explicit hold requires operator resolution')
        watcher=None if config.get('notify')=='stdout' else Path(config['watcher_state']);count=0
        while not (state_dir/'STOP').exists():
            try:
                current=snapshot(config);now=time.time()
                event=observe(state,current,now)
                if event:
                    # Save before notification: uncertain delivery is never blindly repeated.
                    state['last_event']=event;state['notification']='prepared';write(cache,state)
                    if config.get('notify')=='stdout':notify_stdout(event,current)
                    else:notify(state_dir,watcher,event)
                    state['notification']='requested'
                write(cache,state)
                delay=60 if state.get('candidate_hash') else 120
                status={'pid':os.getpid(),'status':'waiting','last_check_utc':datetime.now(timezone.utc).isoformat(),
                        'next_check_utc':datetime.fromtimestamp(time.time()+delay,timezone.utc).isoformat(),
                        'interval_seconds':delay,'entries':len(current),'baseline_sha256':state['baseline_hash'],
                        'last_event':state.get('last_event',{}).get('revision')}
                write(state_dir/'status.json',status)
            except Exception as error:
                # No secret-bearing exception text/HTTP response is published.
                failure={'pid':os.getpid(),'status':'hold','utc':datetime.now(timezone.utc).isoformat(),
                         'error_type':type(error).__name__,'remote_status_verified':False}
                if isinstance(error,PollError):
                    failure['reason']=error.reason
                    if error.http_status is not None:failure['http_status']=error.http_status
                write(state_dir/'HOLD',failure);write(state_dir/'status.json',failure)
                if config.get('notify')=='stdout':
                    print('DROPBOX_WATCH_HOLD reason=%s http=%s utc=%s'%(failure.get('reason',failure['error_type']),
                          failure.get('http_status'),failure['utc']),flush=True)
                # Existing30min watcher will see HOLD; no retry/probe/write to remote.
                return 1
            count+=1
            if stop_after is not None and count>=stop_after:return 0
            until=time.monotonic()+delay
            while time.monotonic()<until and not (state_dir/'STOP').exists():time.sleep(min(1,max(0,until-time.monotonic())))
        write(state_dir/'status.json',{'status':'stopped','utc':datetime.now(timezone.utc).isoformat()})
        return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--stdout',action='store_true',help='print stable changes/HOLD to stdout instead of queue-watcher wake')
    args=parser.parse_args();os.umask(0o077)
    config=json.loads(args.config.read_text())
    if args.stdout:config['notify']='stdout'
    sys.exit(run(config,args.state.resolve()))


if __name__=='__main__':main()
