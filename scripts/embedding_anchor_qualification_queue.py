"""Two bounded anchor qualification panels; unchanged native handoff helpers."""
import argparse,json,os,signal,sys,time,traceback
from pathlib import Path
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.embedding_anchor_qualification_finish import finish
from scripts.gpu_process import inspect,process,ownership
from scripts.gpu_status import require_free
from scripts.train_then_burn import stop_burns,enable_subreaper,clean_owned_children,GUARD_HASH
from scripts.warm_order_queue import check_source,spawn,native,NATIVE,NATIVE_SHA

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args();base=Path(a.base);check_source(base)
    admission=json.loads((base/'execution/admission.json').read_text());expected=admission['ownership']
    def unchanged():
        current=inspect();return ownership(current['gpus'])==ownership(expected['gpus']) and all(process(int(pid))==r for pid,r in expected['processes'].items())
    if not unchanged():raise ValueError('Ownership changed, no signals')
    guard=Path('/mnt/local/_gpu_guard/DISABLED')
    if sha(guard.parent/'gpu_guard.sh')!=GUARD_HASH:raise ValueError('Guard changed')
    marker=json.dumps(dict(owner_pid=os.getpid(),base=str(base),utc=time.time()))
    with guard.open('x') as f:f.write(marker)
    inode=guard.stat().st_ino;time.sleep(30)
    if not unchanged():
        if guard.exists() and guard.stat().st_ino==inode and guard.read_text()==marker:guard.unlink()
        raise ValueError('Ownership changed during hold, no signals')
    enable_subreaper()
    def interrupted(signum,frame):raise KeyboardInterrupt('Owned anchor qualification interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        stop_burns(base/'execution',{NATIVE:NATIVE_SHA});plan=json.loads((base/'execution/plan.json').read_text());active=[];failures=[]
        for seed in (0,1):
            args=[sys.executable,'-u','-m','efficient_llm.warm_order.embedding_anchor_qualify','--prepared',plan['prepared_root'],'--sha',plan['prepared_sha256'],'--seed',str(seed),'--out',str(base/f's{seed}')]
            active.append(spawn(base,'study',f's{seed}',seed,args))
        while active:
            write_json(base/'status.json',dict(stage='study',active=[{k:v for k,v in j.items() if k!='child'} for j in active],failures=failures,utc=time.time()))
            for job in list(active):
                code=job['child'].poll()
                if code is not None:
                    active.remove(job)
                    if code:failures.append(dict(name=job['name'],code=code))
            if active:time.sleep(10)
        if failures:raise ValueError('Anchor qualification failures '+repr(failures))
        finish(base);write_json(base/'qualification-complete.json',dict(complete=True,utc=time.time()))
    except BaseException as error:
        write_json(base/'failure.json',dict(error=repr(error),utc=time.time()));traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        clean_owned_children();time.sleep(30);require_free(list(range(8)));native(base,session='embedding_anchor_gpu_20261122_native_a01')
        if guard.exists() and guard.stat().st_ino==inode and guard.read_text()==marker:guard.unlink()
        else:raise ValueError('Guard ownership changed')
        success=(base/'qualification-complete.json').exists();write_json(base/'terminal.json',dict(success=success,native_verified=True,utc=time.time()));write_json(base/'status.json',dict(stage='terminal',active=[],success=success,utc=time.time()))
if __name__=='__main__':main()
