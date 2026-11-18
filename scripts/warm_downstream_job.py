"""Authorized frozen-checkpoint downstream pipeline; no training code is invoked."""
import argparse,json,os,shlex,shutil,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.train import write_json
from efficient_llm.warm_order.downstream import load,score,prepare_rows
from efficient_llm.warm_order.downstream_report import compact,report,TASKS
from scripts.gpu_process import inspect,process,ownership
from scripts.gpu_status import require_free
from scripts.train_then_burn import stop_burns,enable_subreaper,clean_owned_children,approved_launcher,GUARD_HASH
from scripts.warm_order_queue import check_source,spawn,native,NATIVE,NATIVE_SHA

BASE=Path('/mnt/local/_outputs/deep-llms_th2/warm_downstream_20261118')
TOKENIZER='/mnt/local/_data/deep-llms_th2/a2_dot_20261118/tokenizer'
DATA='/mnt/local/_data/deep-llms_th2/warm_downstream_20261118'
EVAL='/mnt/local/conda-py311/envs/eval/bin/python3.11'
MANIFEST='resources/warm_downstream_data_20261118.json'


def bridge(mode,output,*extra):
    argv=[EVAL,'-m','eval.a2_requests',mode,'--tokenizer',TOKENIZER,'--dataset-root',DATA,'--manifest',MANIFEST,'--tasks',*TASKS,'--max-length','512','--output',str(output),*extra]
    with Path(str(output)+'.log').open('x') as log:subprocess.run(argv,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,check=True)


def launch(commit):
    if len(commit)!=40 or socket.gethostname()!='thiennh-p6-ohmv-worker-0':raise ValueError('Wrong commit/node')
    if BASE.exists():raise ValueError('Existing downstream root; retrieve not relaunch')
    pin=json.loads(Path('resources/warm_downstream_source_20261118.json').read_text())
    for rel,w in pin['files'].items():
        if sha(rel)!=w:raise ValueError('Source mismatch '+rel)
    for r in json.loads(Path(MANIFEST).read_text())['repositories']:
        for f in r['files']:
            path=Path(DATA)/f['path']
            if path.stat().st_size!=f['bytes'] or sha(path)!=f['sha256']:raise ValueError('Dataset mismatch')
    expected=json.loads(Path('resources/warm_downstream_models_20261118.json').read_text());models=[]
    for e in expected['models']:
        for p,w in e['hashes'].items():
            if sha(p)!=w:raise ValueError('Checkpoint mismatch '+p)
        models.append(e)
    if torch.__version__!='2.14.1+cu130' or np.__version__!='2.4.6':raise ValueError('Runtime mismatch')
    current=inspect();guard=Path('/mnt/local/_gpu_guard')
    if guard.joinpath('DISABLED').exists() or sha(guard/'gpu_guard.sh')!=GUARD_HASH:raise ValueError('Guard not available')
    if len(current['workers'])!=8 or any(not approved_launcher(process(p)['ppid'],{NATIVE:NATIVE_SHA}) for p in current['workers']):raise ValueError('Unverified GPU owner')
    if (guard/'burn.pids').exists() and (guard/'burn.pids').read_text().strip():raise ValueError('Unexpected guard-owned workers')
    BASE.mkdir();execution=BASE/'execution';execution.mkdir();frozen=execution/'source'
    for rel,w in pin['files'].items():
        dst=frozen/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(rel,dst)
        if sha(dst)!=w:raise ValueError('Freeze mismatch')
    write_json(execution/'source.json',pin['files']);write_json(execution/'admission.json',dict(commit=commit,ownership=current,utc=time.time(),models=models,manifest_sha256=sha(MANIFEST)))
    write_json(BASE/'plan.json',dict(models=models,requests=str(BASE/'requests.json'),batch_size=16))
    env=['CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','HF_HUB_OFFLINE=1','HF_DATASETS_OFFLINE=1','TRANSFORMERS_OFFLINE=1','HF_HUB_DISABLE_TELEMETRY=1','WANDB_MODE=offline','TOKENIZERS_PARALLELISM=false','OMP_NUM_THREADS=4','PYTHONPATH='+str(frozen)]
    cmd='exec '+shlex.join(['env',*env,sys.executable,'-u','-m','scripts.warm_downstream_job','run'])+' >'+shlex.quote(str(execution/'queue.log'))+' 2>&1'
    session='warm_downstream_20261118_a01'
    if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Existing session')
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),cmd],check=True)
    subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
    write_json(execution/'submitted.json',dict(commit=commit,session=session,utc=time.time(),source_sha256=sha(execution/'source.json')))
    print('DOWNSTREAM_SUBMITTED',json.dumps(dict(commit=commit,session=session)),flush=True)


def probe():
    plan=json.loads((BASE/'plan.json').read_text());req=json.loads((BASE/'requests.json').read_text())['requests']
    # Fixed spread incl longest and shortest contexts, selected without outcomes.
    ix=sorted(set([0,len(req)//4,len(req)//2,3*len(req)//4,len(req)-1]+sorted(range(len(req)),key=lambda i:len(req[i]['context']))[:3]+sorted(range(len(req)),key=lambda i:-len(req[i]['context']))[:3]))
    small=[req[i] for i in ix];checks={};torch.set_num_threads(4)
    for name in ('pilot-D-s0','pilot-R-s0','local-CF-s0','local-PG-s0'):
        e=next(e for e in plan['models'] if e['name']==name);m,r,o=load(e,'cpu');cpu,cd=score(m,r,small,o,'cpu',1)
        m=m.to('cuda');r=r.to('cuda') if r is not None else None
        one,od=score(m,r,small,o,'cuda',1);batch,bd=score(m,r,small,o,'cuda',16)
        err=max(abs(a[0]-b[0])/len(q['continuation']) for a,b,q in zip(cpu,one,small));berr=max(abs(a[0]-b[0])/len(q['continuation']) for a,b,q in zip(one,batch,small))
        if err>.03 or berr>.003 or od['choices']!=bd['choices'] or cd['choices']!=od['choices'] or any(a[1]!=b[1] for a,b in zip(one,batch)):raise ValueError('Numerical/batch/route gate failed '+name)
        checks[name]=dict(cpu_gpu_max_nats_per_token=err,batch_max_nats_per_token=berr,choices=od['choices'])
        del m,r;torch.cuda.empty_cache()
    write_json(BASE/'gpu-probe.json',dict(checks=checks,indices=ix,requests=len(small),passed=True))


def replay(name):
    run=BASE/'runs'/name;check_source(BASE);bridge('replay',run/'raw.json','--prepared',str(BASE/'requests.json'),'--scored',str(run/'scores.json'))
    out=compact(json.loads((run/'raw.json').read_text()),json.loads((BASE/'requests.json').read_text()),json.loads((run/'scores.json').read_text()))
    write_json(run/'items.json',out);write_json(run/'replay-complete.json',dict(items_sha256=sha(run/'items.json'),raw_sha256=sha(run/'raw.json'),tasks={t:dict(accuracy=v['accuracy'],items=len(v['items'])) for t,v in out.items()}))


def run():
    check_source(BASE)
    # CPU input preparation while the existing native workload keeps its GPUs.
    try:
        subprocess.run([sys.executable,'-m','unittest','tests.warm_order.test_downstream','-v'],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True)
        write_json(BASE/'status.json',dict(stage='cpu_prepare',utc=time.time()));bridge('prepare',BASE/'requests.json')
        plan=json.loads((BASE/'plan.json').read_text());requests=json.loads((BASE/'requests.json').read_text());prepare_rows(requests['requests'])
        plan['requests_sha256']=sha(BASE/'requests.json');write_json(BASE/'plan.json',plan)
    except BaseException as e:
        write_json(BASE/'failure.json',dict(stage='cpu_prepare',error=repr(e),utc=time.time()));raise
    expected=json.loads((BASE/'execution/admission.json').read_text())['ownership']
    def unchanged():
        now=inspect();return ownership(now['gpus'])==ownership(expected['gpus']) and all(process(int(pid))==r for pid,r in expected['processes'].items())
    if not unchanged():raise ValueError('Ownership changed before hold')
    guard=Path('/mnt/local/_gpu_guard/DISABLED');marker=json.dumps(dict(owner_pid=os.getpid(),base=str(BASE),utc=time.time()))
    with guard.open('x') as f:f.write(marker)
    inode=guard.stat().st_ino;time.sleep(30)
    if not unchanged():raise ValueError('Ownership changed during hold; no signals')
    enable_subreaper();stop_burns(BASE/'execution',{NATIVE:NATIVE_SHA})
    def interrupted(s,f):raise KeyboardInterrupt('Owned downstream interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    active=[];pending=list(plan['models']);failed=[];cpu=[]
    try:
        j=spawn(BASE,'probe','numerical',0,[sys.executable,'-u','-m','scripts.warm_downstream_job','probe'])
        if j['child'].wait()!=0:raise ValueError('GPU numerical gate failed')
        while pending or active or cpu:
            occupied={j['gpu'] for j in active}
            for gpu in range(8):
                if pending and gpu not in occupied:
                    e=pending.pop(0);active.append(spawn(BASE,'score',e['name'],gpu,[sys.executable,'-u','-m','efficient_llm.warm_order.downstream','--plan',str(BASE/'plan.json'),'--name',e['name'],'--out',str(BASE/'runs'/e['name'])]))
            for j in list(active):
                code=j['child'].poll()
                if code is None:continue
                active.remove(j)
                if code:failed.append((j['name'],'score',code));continue
                log=(BASE/f"replay-{j['name']}.log").open('x');q=subprocess.Popen([sys.executable,'-u','-m','scripts.warm_downstream_job','replay','--name',j['name']],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT);log.close();cpu.append(dict(name=j['name'],child=q,pid=q.pid))
            for j in list(cpu):
                code=j['child'].poll()
                if code is not None:
                    cpu.remove(j)
                    if code:failed.append((j['name'],'replay',code))
            write_json(BASE/'status.json',dict(stage='eval_replay',pending=[e['name'] for e in pending],active=[{k:v for k,v in j.items() if k!='child'} for j in active],replays=[j['name'] for j in cpu],failures=failed,utc=time.time()))
            if pending or active or cpu:time.sleep(5)
        if failed:raise ValueError('Evaluation/replay failed '+repr(failed))
        check_source(BASE)
        for e in plan['models']:
            for p,w in e['hashes'].items():
                if sha(p)!=w:raise ValueError('Checkpoint changed during eval')
        report(BASE);write_json(BASE/'scientific-complete.json',dict(success=True,utc=time.time()))
    except BaseException as e:
        write_json(BASE/'failure.json',dict(error=repr(e),utc=time.time()));traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN);clean_owned_children();time.sleep(30);require_free(list(range(8)))
        native(BASE,session='warm_downstream_20261118_native_a01')
        if guard.exists() and guard.stat().st_ino==inode and guard.read_text()==marker:guard.unlink()
        else:raise ValueError('Guard owner changed')
        write_json(BASE/'terminal.json',dict(success=(BASE/'scientific-complete.json').exists(),native_verified=True,utc=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['launch','run','probe','replay']);p.add_argument('--commit');p.add_argument('--name');a=p.parse_args()
    if a.mode=='launch':launch(a.commit)
    elif a.mode=='run':run()
    elif a.mode=='probe':probe()
    else:replay(a.name)
if __name__=='__main__':main()
