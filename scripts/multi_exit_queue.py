"""Frozen token-order router fits; preserve all prior studies; verified backup/native handoff."""
import argparse,json,os,signal,socket,sys,time,traceback
from pathlib import Path
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.train import write_json
from efficient_llm.warm_order.multi_exit_report import NAMES,report
from scripts.gpu_process import inspect,process,ownership
from scripts.gpu_status import require_free
from scripts.train_then_burn import stop_burns,enable_subreaper,clean_owned_children,GUARD_HASH
from scripts.warm_order_queue import check_source,spawn,native,NATIVE,NATIVE_SHA


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args();base=Path(a.base)
    if socket.gethostname()!='thiennh-p6-ohmv-worker-0':raise ValueError('Wrong node')
    check_source(base);expected=json.loads((base/'execution/admission.json').read_text())['ownership']
    def unchanged():
        current=inspect()
        return ownership(current['gpus'])==ownership(expected['gpus']) and all(process(int(pid))==r for pid,r in expected['processes'].items())
    if not unchanged():raise ValueError('GPU ownership changed; no signals')
    guard=Path('/mnt/local/_gpu_guard/DISABLED')
    if sha(guard.parent/'gpu_guard.sh')!=GUARD_HASH:raise ValueError('Guard protocol changed')
    marker=json.dumps(dict(owner_pid=os.getpid(),base=str(base),utc=time.time()))
    with guard.open('x') as f:f.write(marker)
    inode=guard.stat().st_ino;time.sleep(30)
    if not unchanged():raise ValueError('Ownership changed during hold; no signals')
    enable_subreaper();stop_burns(base/'execution',{NATIVE:NATIVE_SHA})
    def interrupted(signum,frame):raise KeyboardInterrupt('Owned queue interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    active=[];failures=[]
    try:
        pending=list(NAMES)
        while active or pending:
            free=[g for g in range(8) if g not in {j['gpu'] for j in active}]
            for gpu in free:
                if not pending or failures:break
                name=pending.pop(0)
                active.append(spawn(base,'fit_eval',name,gpu,[sys.executable,'-u','-m','efficient_llm.warm_order.multi_exit_eval','--config',str(base/f'inputs/{name}.json'),'--out',str(base/f'runs/{name}')]))
            write_json(base/'status.json',dict(stage='fit_eval',active=[{k:v for k,v in x.items() if k!='child'} for x in active],failures=failures,utc=time.time()))
            for job in list(active):
                code=job['child'].poll()
                if code is None:continue
                active.remove(job)
                if code!=0:failures.append(dict(name=job['name'],code=code))
            if failures:pending=[]
            if active or pending:time.sleep(10)
        if failures:raise ValueError('Project jobs failed: '+repr(failures))
        report(base,Path('/mnt/local/_backup/deep-llms_th2/multi_exit_20261119_a01'))
        write_json(base/'scientific-complete.json',dict(ready=True,utc=time.time(),backup_sha256=sha(base/'backup.json')))
    except BaseException as e:
        write_json(base/'failure.json',dict(error=repr(e),utc=time.time()));traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        clean_owned_children();time.sleep(30);require_free(list(range(8)))
        native(base,session='multi_exit_20261119_native_a01')
        if guard.exists() and guard.stat().st_ino==inode and guard.read_text()==marker:guard.unlink()
        else:raise ValueError('Guard owner changed; refuse removal')
        success=(base/'scientific-complete.json').exists()
        write_json(base/'status.json',dict(stage='terminal',active=[],success=success,utc=time.time()))
        write_json(base/'terminal.json',dict(success=success,native_verified=True,utc=time.time()))
if __name__=='__main__':main()
