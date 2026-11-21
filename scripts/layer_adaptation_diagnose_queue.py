"""Only four bounded numerical/resume jobs, with verified native handoff."""
import argparse,json,os,signal,socket,sys,time,traceback
from pathlib import Path
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.adaptation_train import source_hashes
from efficient_llm.warm_order.adaptation_runtime import training_policy
from scripts.gpu_process import inspect,process,ownership
from scripts.gpu_status import require_free
from scripts.train_then_burn import stop_burns,enable_subreaper,clean_owned_children,GUARD_HASH
from scripts.warm_order_queue import check_source,spawn,native,NATIVE,NATIVE_SHA

NAMES=('A-s0','B-s0','A-s1','B-s1')

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
        argv=[sys.executable,'-u','-m','efficient_llm.warm_order.adaptation_numerical_diagnose',
              '--config',str(prepared/'B-s1.json'),'--out',str(base/'diagnosis')]
        job=spawn(base,'inference','B-s1',3,argv);active.append(job)
        while job['child'].poll() is None:
            write_json(base/'status.json',dict(stage='inference_diagnosis',active=[{k:v for k,v in job.items() if k!='child'}],utc=time.time()))
            time.sleep(5)
        if job['child'].returncode:raise ValueError('Numerical diagnosis failed; inspect exact project log')
        done=json.loads((base/'diagnosis/complete.json').read_text())
        if done['training_updates']!=0 or sha(base/'diagnosis/REPORT.json')!=done['report_sha256']:raise ValueError('Diagnosis proof mismatch')
        write_json(base/'qualification-complete.json',dict(inference_only=True,report_sha256=done['report_sha256'],utc=time.time()))
    except BaseException as error:
        write_json(base/'failure.json',dict(error=repr(error),utc=time.time()));traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        clean_owned_children();time.sleep(30);require_free(list(range(8)))
        native(base,session='layer_adaptation_diagnose_20261121_native_a01')
        if guard.exists() and guard.stat().st_ino==inode and guard.read_text()==marker:guard.unlink()
        else:raise ValueError('Guard owner changed; refuse removal')
        success=(base/'qualification-complete.json').exists()
        write_json(base/'terminal.json',dict(success=success,native_verified=True,utc=time.time()))
        write_json(base/'status.json',dict(stage='terminal',active=[],success=success,utc=time.time()))
if __name__=='__main__':main()
