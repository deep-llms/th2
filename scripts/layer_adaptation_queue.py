"""Owned four-branch paired continuation, two studies and verified backup/native handoff."""
import argparse,json,os,signal,socket,sys,time,traceback
from pathlib import Path
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.adaptation_finish import finish,NAMES
from scripts.gpu_process import inspect,process,ownership
from scripts.gpu_status import require_free
from scripts.train_then_burn import stop_burns,enable_subreaper,clean_owned_children,GUARD_HASH
from scripts.warm_order_queue import check_source,spawn,native,NATIVE,NATIVE_SHA

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args();base=Path(a.base)
    if socket.gethostname()!='thiennh-p6-ohmv-worker-0':raise ValueError('Wrong node')
    check_source(base);admission=json.loads((base/'execution/admission.json').read_text());expected=admission['ownership']
    def unchanged():
        current=inspect()
        return ownership(current['gpus'])==ownership(expected['gpus']) and all(process(int(pid))==r for pid,r in expected['processes'].items())
    if not unchanged():raise ValueError('Ownership changed; no signals')
    guard=Path('/mnt/local/_gpu_guard/DISABLED')
    if sha(guard.parent/'gpu_guard.sh')!=GUARD_HASH:raise ValueError('Guard protocol changed')
    marker=json.dumps(dict(owner_pid=os.getpid(),base=str(base),utc=time.time()))
    with guard.open('x') as f:f.write(marker)
    inode=guard.stat().st_ino;time.sleep(30)
    if not unchanged():
        if guard.exists() and guard.stat().st_ino==inode and guard.read_text()==marker:guard.unlink()
        raise ValueError('Ownership changed during hold; no signals')
    enable_subreaper();active=[];failures=[]
    def interrupted(signum,frame):raise KeyboardInterrupt('Owned qualification interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        stop_burns(base/'execution',{NATIVE:NATIVE_SHA})
        prepared=Path(admission['prepared']).parent
        def stage(stage_name,specs):
            active=[];failures=[]
            for gpu,name,argv in specs:active.append(spawn(base,stage_name,name,gpu,argv))
            while active:
                write_json(base/'status.json',dict(stage=stage_name,active=[{k:v for k,v in x.items() if k!='child'} for x in active],failures=failures,utc=time.time()))
                for job in list(active):
                    code=job['child'].poll()
                    if code is not None:
                        active.remove(job)
                        if code:failures.append(dict(name=job['name'],code=code))
                if active:time.sleep(10)
            if failures:raise ValueError('Project '+stage_name+' failures: '+repr(failures))
        specs=[]
        for gpu,name in enumerate(NAMES):
            argv=[sys.executable,'-u','-m','efficient_llm.warm_order.adaptation_run','train',
                  '--prepared',str(prepared),'--prepared-sha256',admission['prepared_sha256'],
                  '--gate',admission['gate'],'--gate-sha256',admission['gate_sha256'],
                  '--name',name,'--out',str(base/'runs'/name)]
            specs.append((gpu,name,argv))
        stage('continuation',specs)
        stage('study',[(seed,'s'+str(seed),[sys.executable,'-u','-m','efficient_llm.warm_order.adaptation_run',
               'study','--base',str(base),'--seed',str(seed),'--out',str(base/'study'/('s'+str(seed)))]) for seed in (0,1)])
        proof=finish(base,Path('/mnt/local/_backup/deep-llms_th2/layer_adaptation_20261121_a01'))
        write_json(base/'scientific-complete.json',dict(ready=True,backup_sha256=sha(base/'backup.json'),utc=time.time()))
    except BaseException as error:
        write_json(base/'failure.json',dict(error=repr(error),utc=time.time()));traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        clean_owned_children();time.sleep(30);require_free(list(range(8)))
        native(base,session='layer_adaptation_20261121_native_a01')
        if guard.exists() and guard.stat().st_ino==inode and guard.read_text()==marker:guard.unlink()
        else:raise ValueError('Guard owner changed; refuse removal')
        success=(base/'scientific-complete.json').exists()
        write_json(base/'terminal.json',dict(success=success,native_verified=True,utc=time.time()))
        write_json(base/'status.json',dict(stage='terminal',active=[],success=success,utc=time.time()))
if __name__=='__main__':main()
