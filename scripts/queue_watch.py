"""Wake one existing conversation, with optional one-shot adaptive scheduling.

Local-only; no API keys, new agents, GPU management, or remote shell access.
Queues a message on the existing app server; never opens a competing writer. Ends at a fixed deadline or STOP; an explicitly SHA-locked user policy can remove the deadline.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def scheduled_wake(path, seen, now):
    """Consume a revision once; a short wake does not become a rapid repeat loop."""
    try:
        data=json.loads(path.read_text())
        revision=data['revision']
        if not isinstance(revision,str) or not revision:raise ValueError('revision required')
        if revision==seen:return None
        due=datetime.fromisoformat(data['next_utc'])
        if due.tzinfo is None:raise ValueError('next_utc timezone required')
        message=data.get('message')
        if message is not None and (not isinstance(message,str) or not 1<=len(message)<=4000):
            raise ValueError('Optional message must contain1–4000characters')
        # No requested wake may defer checking beyond thirty minutes.
        return revision,max(now,min(due.timestamp(),now+1800)),message
    except FileNotFoundError:
        return None


def deadline(until, policy=None, policy_sha=None):
    if policy is None:
        if not until or policy_sha:raise ValueError('Deadline or explicit policy required')
        value=datetime.fromisoformat(until)
        if value.tzinfo is None:raise ValueError('Deadline timezone required')
        return value.timestamp()
    if until or policy_sha!='e9cf0904bf92c8365b209ce940150cb33ce83c5987512242d3dcb2ef4d79a3ac':
        raise ValueError('Explicit report-only time policy required')
    raw=Path(policy).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=policy_sha:raise ValueError('Time policy changed')
    value=json.loads(raw)
    if (value.get('time_limits_report_only') is not True or value.get('continue_registered_work') is not True
            or value.get('hard_deadline') is not None):raise ValueError('Policy does not authorize continuation')
    return float('inf')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--thread',required=True)
    p.add_argument('--cwd',type=Path,required=True)
    p.add_argument('--prompt',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--interval',type=int,default=1800)
    stop=p.add_mutually_exclusive_group(required=True)
    stop.add_argument('--until')
    stop.add_argument('--time-policy',type=Path)
    p.add_argument('--time-policy-sha256')
    p.add_argument('--first-at')
    p.add_argument('--codex',default='/home/users/thien/.local/bin/codex')
    a=p.parse_args()
    if not 30<=a.interval<=1800:raise ValueError('Interval must be30–1800seconds')
    end=deadline(a.until,a.time_policy,a.time_policy_sha256)
    os.umask(0o077);a.state.mkdir(parents=True,exist_ok=True)
    with (a.state/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Let the current user-facing turn finish before resuming this same thread.
        next_run=datetime.fromisoformat(a.first_at).timestamp() if a.first_at else time.time()+a.interval
        previous=json.loads((a.state/'status.json').read_text()) if (a.state/'status.json').exists() else {}
        tick=previous.get('tick',0)
        last_delivery=previous.get('last_delivery')
        schedule_revision=previous.get('schedule_revision')
        scheduled_message=previous.get('scheduled_message')
        schedule_error=None
        next_run=min(next_run,time.time()+1800)
        while time.time()<end and not (a.state/'STOP').exists():
            status={'pid':os.getpid(),'thread':a.thread,'tick':tick,'status':'waiting',
                    'next_utc':datetime.fromtimestamp(next_run,timezone.utc).isoformat(),
                    'until':a.until,'time_policy':str(a.time_policy) if a.time_policy else None,
                    'time_policy_sha256':a.time_policy_sha256,
                    'interval_seconds':a.interval,'schedule_revision':schedule_revision,
                    'schedule_error':schedule_error,
                    'scheduled_message':scheduled_message,
                    'last_delivery':last_delivery}
            write(a.state/'status.json',status)
            while time.time()<end and not (a.state/'STOP').exists():
                old_error=schedule_error
                try:
                    change=scheduled_wake(a.state/'schedule.json',schedule_revision,time.time())
                    schedule_error=None
                except (ValueError,KeyError,TypeError,OverflowError,OSError) as error:
                    change=None
                    schedule_error=f'{type(error).__name__}: invalid local schedule; previous wake retained'
                if change is not None:
                    schedule_revision,next_run,scheduled_message=change
                if change is not None or old_error!=schedule_error:
                    status.update(next_utc=datetime.fromtimestamp(next_run,timezone.utc).isoformat(),
                                  schedule_revision=schedule_revision,schedule_error=schedule_error,
                                  scheduled_message=scheduled_message)
                    write(a.state/'status.json',status)
                if time.time()>=next_run:break
                time.sleep(1)
            if time.time()>=end or (a.state/'STOP').exists():break
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            log=a.state/(stamp+'.log')
            argv=[a.codex,'queue','--remote','unix://','--resume','--thread',a.thread,
                  '--message',scheduled_message or a.prompt.read_text()]
            status.update(status='running',started_utc=stamp,log=str(log));write(a.state/'status.json',status)
            # App server queues behind an active turn; preserve its model and permissions.
            with log.open('x') as handle:
                try:
                    result=subprocess.run(argv,text=True,cwd=a.cwd,
                                          stdout=handle,stderr=subprocess.STDOUT,timeout=60)
                    returncode=result.returncode
                except subprocess.TimeoutExpired:
                    returncode=124
                    handle.write('\nWatcher: queue command timed out; delivery is unconfirmed.\n')
                except OSError as error:
                    returncode=127
                    handle.write(f'\nWatcher: could not execute queue command: {error}\n')
            status.update(status='queued' if returncode == 0 else 'delivery_failed',returncode=returncode)
            last_delivery={'utc':datetime.now(timezone.utc).isoformat(),
                           'status':status['status'],'returncode':returncode,'log':str(log)}
            status['last_delivery']=last_delivery
            write(a.state/(stamp+'.json'),status)
            tick+=1
            scheduled_message=None
            # A one-shot early wake falls back to the normal interval.
            next_run=time.time()+a.interval
        write(a.state/'status.json',{'status':'stopped','tick':tick,'thread':a.thread,
                                    'last_delivery':last_delivery,'schedule_revision':schedule_revision,
                                    'utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':main()
