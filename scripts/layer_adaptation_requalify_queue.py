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

def validate_record(directory,config,admission,expected_complete=None):
    directory=Path(directory)
    if expected_complete is not None and sha(directory/'complete.json')!=expected_complete:
        raise ValueError('Reused complete receipt changed')
    r=json.loads((directory/'complete.json').read_text());c=config
    if (r['branch']!=c['branch'] or r['seed']!=c['row_reservation']['seed'] or r['parent_hashes']!=c['parent_hashes'] or
        r['source']!=source_hashes() or r.get('training_policy')!=training_policy('cuda') or
        r['prepared_sha256']!=admission['prepared_sha256'] or r['reservation_sha256']!=c['row_reservation']['manifest_sha256'] or
        r['step']!=40 or not all(r[k] is True for k in ('exact_resume','finite','parent_step0_exact','native_component_parity_pass'))):
        raise ValueError('Gate identity/policy mismatch')
    for sub,key in [('whole','whole_fullstate_sha256'),('split','split_fullstate_sha256')]:
        if sha(directory/sub/'fullstate.pt')!=r[key]:raise ValueError('Gate checkpoint changed')
    if sha(directory/'numerical.json')!=r['numerical_sha256']:raise ValueError('Numerical evidence changed')
    first=(directory/'whole/trace.jsonl').read_bytes()
    if first!=(directory/'split/trace.jsonl').read_bytes() or [json.loads(l)['step'] for l in first.splitlines()]!=list(range(1,41)):
        raise ValueError('Exact first40 traces changed')
    return r


def reused_records(admission):
    registration=json.loads(Path('resources/layer_adaptation_reused_gates_20261121.json').read_text())['branches']
    if set(registration)!={'A-s0','B-s0','A-s1'}:raise ValueError('Exact three prior passed branches required')
    prepared=Path(admission['prepared']).parent;records={}
    for name,pin in registration.items():
        c=json.loads((prepared/(name+'.json')).read_text())
        r=validate_record(pin['directory'],c,admission,pin['complete_sha256'])
        if (r['whole_fullstate_sha256']!=pin['whole_sha256'] or r['split_fullstate_sha256']!=pin['split_sha256'] or
            r['numerical_sha256']!=pin['numerical_sha256']):raise ValueError('Reused byte pin mismatch')
        records[name]=r
    return records


def aggregate(base,admission):
    records=reused_records(admission);prepared=Path(admission['prepared']).parent
    records['B-s1']=validate_record(base/'runs/B-s1',json.loads((prepared/'B-s1.json').read_text()),admission)
    signatures={(r['torch'],r['precision'],r['tf32'],r['reservation_sha256']) for r in records.values()}
    if len(signatures)!=1:raise ValueError('Mixed gate runtimes/reservations')
    runtime=signatures.pop()
    report=dict(source=source_hashes(),training_policy=training_policy('cuda'),torch=runtime[0],precision=runtime[1],tf32=runtime[2],
        reservation_sha256=runtime[3],native_component_parity_pass=True,branches=records,
        new_qualification_updates_executed=80,reused_qualification_updates=240,qualification_updates_executed=320,
        production_updates_executed=0,reused_registration_sha256=sha('resources/layer_adaptation_reused_gates_20261121.json'),
        scope='B-s1 requalification after independent same-row/GPU repeated inference passed; a02 failed forward retained, cause not established.')
    write_json(base/'gpu-admission.json',report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args();base=Path(a.base)
    if socket.gethostname()!='thiennh-p6-ohmv-worker-0':raise ValueError('Wrong node')
    check_source(base);admission=json.loads((base/'execution/admission.json').read_text());reused_records(admission);expected=admission['ownership']
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
        for gpu,name in [(3,'B-s1')]:
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
        native(base,session='layer_adaptation_gate_20261121_native_a03')
        if guard.exists() and guard.stat().st_ino==inode and guard.read_text()==marker:guard.unlink()
        else:raise ValueError('Guard owner changed; refuse removal')
        success=(base/'qualification-complete.json').exists()
        write_json(base/'terminal.json',dict(success=success,native_verified=True,utc=time.time()))
        write_json(base/'status.json',dict(stage='terminal',active=[],success=success,utc=time.time()))
if __name__=='__main__':main()
