"""Only four bounded numerical/resume jobs, with verified native handoff."""
import argparse,json,os,signal,socket,sys,time,traceback
from pathlib import Path
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.adaptation_train import source_hashes
from scripts.gpu_process import inspect,process,ownership
from scripts.gpu_status import require_free
from scripts.train_then_burn import stop_burns,enable_subreaper,clean_owned_children,GUARD_HASH
from scripts.warm_order_queue import check_source,spawn,native,NATIVE,NATIVE_SHA

NAMES=('A-s0','B-s0','A-s1','B-s1')

def aggregate(base,admission):
    prepared=Path(admission['prepared']).parent;branches={};reservation=None;runtime=None
    for name in NAMES:
        c=json.loads((prepared/(name+'.json')).read_text());r=json.loads((base/'runs'/name/'complete.json').read_text())
        if (r['branch']!=c['branch'] or r['seed']!=c['row_reservation']['seed'] or r['parent_hashes']!=c['parent_hashes'] or
            r['source']!=source_hashes() or r['prepared_sha256']!=admission['prepared_sha256'] or
            r['step']!=40 or not all(r[k] is True for k in ('exact_resume','finite','parent_step0_exact','native_component_parity_pass'))):raise ValueError('Invalid gate '+name)
        for sub,key in [('whole','whole_fullstate_sha256'),('split','split_fullstate_sha256')]:
            if sha(base/'runs'/name/sub/'fullstate.pt')!=r[key]:raise ValueError('Gate checkpoint changed')
        if sha(base/'runs'/name/'numerical.json')!=r['numerical_sha256']:raise ValueError('Numerical evidence changed')
        sig=(r['torch'],r['precision'],r['tf32'],r['reservation_sha256'])
        if runtime is None:runtime=sig
        elif sig!=runtime:raise ValueError('Mixed gate runtime/reservation')
        branches[name]=r
    report=dict(source=source_hashes(),torch=runtime[0],precision=runtime[1],tf32=runtime[2],reservation_sha256=runtime[3],
                native_component_parity_pass=True,branches=branches,qualification_updates_executed=320,
                production_updates_executed=0,scope='Numerical/fullstate qualification only; no causal mechanism conclusion')
    write_json(base/'gpu-admission.json',report)
    return report

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
    if not unchanged():raise ValueError('Ownership changed during hold; no signals')
    enable_subreaper();active=[];failures=[]
    def interrupted(signum,frame):raise KeyboardInterrupt('Owned qualification interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        stop_burns(base/'execution',{NATIVE:NATIVE_SHA})
        prepared=Path(admission['prepared']).parent
        for gpu,name in enumerate(NAMES):
            argv=[sys.executable,'-u','-m','efficient_llm.warm_order.adaptation_gpu_probe','--config',str(prepared/(name+'.json')),
                  '--prepared',admission['prepared'],'--prepared-sha256',admission['prepared_sha256'],'--out',str(base/'runs'/name)]
            active.append(spawn(base,'qualify',name,gpu,argv))
        while active:
            write_json(base/'status.json',dict(stage='qualification',active=[{k:v for k,v in x.items() if k!='child'} for x in active],failures=failures,utc=time.time()))
            for job in list(active):
                code=job['child'].poll()
                if code is not None:
                    active.remove(job)
                    if code:failures.append(dict(name=job['name'],code=code))
            if active:time.sleep(5)
        if failures:raise ValueError('Qualification failures: '+repr(failures))
        aggregate(base,admission)
        write_json(base/'qualification-complete.json',dict(ready=True,gpu_admission_sha256=sha(base/'gpu-admission.json'),utc=time.time()))
    except BaseException as error:
        write_json(base/'failure.json',dict(error=repr(error),utc=time.time()));traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        clean_owned_children();time.sleep(30);require_free(list(range(8)))
        native(base,session='layer_adaptation_gate_20261121_native_a01')
        if guard.exists() and guard.stat().st_ino==inode and guard.read_text()==marker:guard.unlink()
        else:raise ValueError('Guard owner changed; refuse removal')
        success=(base/'qualification-complete.json').exists()
        write_json(base/'terminal.json',dict(success=success,native_verified=True,utc=time.time()))
        write_json(base/'status.json',dict(stage='terminal',active=[],success=success,utc=time.time()))
if __name__=='__main__':main()
