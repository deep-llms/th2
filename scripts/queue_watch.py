"""Wake the explicitly selected existing Codex conversation at a fixed interval.

Local-only; no API keys, new agents, GPU management, or remote shell access.
Queues a message on the existing app server; never opens a competing writer. Ends at the fixed deadline or STOP.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--thread',required=True)
    p.add_argument('--cwd',type=Path,required=True)
    p.add_argument('--prompt',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--interval',type=int,default=1800)
    p.add_argument('--until',required=True)
    p.add_argument('--first-at')
    p.add_argument('--codex',default='/home/users/thien/.local/bin/codex')
    a=p.parse_args()
    if a.interval<1800:raise ValueError('Minimum interval30min')
    end=datetime.fromisoformat(a.until).timestamp()
    os.umask(0o077);a.state.mkdir(parents=True,exist_ok=True)
    with (a.state/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Let the current user-facing turn finish before resuming this same thread.
        next_run=datetime.fromisoformat(a.first_at).timestamp() if a.first_at else time.time()+a.interval
        tick=0
        while time.time()<end and not (a.state/'STOP').exists():
            status={'pid':os.getpid(),'thread':a.thread,'tick':tick,'status':'waiting',
                    'next_utc':datetime.fromtimestamp(next_run,timezone.utc).isoformat(),
                    'until':a.until}
            write(a.state/'status.json',status)
            while time.time()<min(next_run,end) and not (a.state/'STOP').exists():time.sleep(1)
            if time.time()>=end or (a.state/'STOP').exists():break
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            log=a.state/(stamp+'.log')
            argv=[a.codex,'queue','--remote','unix://','--thread',a.thread,
                  '--message',a.prompt.read_text()]
            status.update(status='running',started_utc=stamp,log=str(log));write(a.state/'status.json',status)
            # App server queues behind an active turn; preserve its model and permissions.
            with log.open('x') as handle:
                try:
                    result=subprocess.run(argv,text=True,cwd=a.cwd,
                                          stdout=handle,stderr=subprocess.STDOUT,timeout=60)
                    returncode=result.returncode
                except subprocess.TimeoutExpired:
                    returncode=124
            status.update(status='queued' if returncode == 0 else 'delivery_failed',returncode=returncode)
            write(a.state/(stamp+'.json'),status)
            tick+=1
            next_run=max(next_run+a.interval,time.time()+60)
        write(a.state/'status.json',{'status':'stopped','tick':tick,'thread':a.thread,
                                    'utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':main()
