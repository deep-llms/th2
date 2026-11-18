"""Owned OHMV pilot queue; known native burn reclaim and verified native handoff."""
import argparse,json,os,shlex,shutil,signal,socket,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import numpy as np
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.train import write_json
from efficient_llm.warm_order.evaluate import paired_ci
from scripts.gpu_process import inspect,process,ownership
from scripts.gpu_status import require_free,snapshot
from scripts.train_then_burn import stop_burns,enable_subreaper,clean_owned_children,approved_launcher,GUARD_HASH

NATIVE='/tmp/llm_pretrain_burn.py'
NATIVE_SHA='3cdcc857bd01b096e20a02640fa85f0b8be7607e3c2b22a89a704bbac3650857'

def check_source(base):
    for rel,digest in json.loads((base/'execution/source.json').read_text()).items():
        if sha(Path.cwd()/rel)!=digest:raise ValueError('Frozen source changed: '+rel)

def spawn(base,stage,name,gpu,argv):
    check_source(base);require_free([gpu])
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4'}
    log=(base/f'{stage}-{name}.log').open('x')
    child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT);log.close()
    return dict(stage=stage,name=name,gpu=gpu,pid=child.pid,child=child,argv=argv)

def check_train(run):
    proof=json.loads((run/'complete.json').read_text())
    if proof['step']!=12000:raise ValueError('Training not terminal')
    for name,key in [('weights.pt','weights_sha256'),('fullstate.pt','fullstate_sha256')]:
        if sha(run/name)!=proof[key]:raise ValueError('Terminal checkpoint changed')
    return proof

def native(base):
    require_free(list(range(8)))
    if sha(NATIVE)!=NATIVE_SHA:raise ValueError('Native script changed')
    session='warm_order_20261118_native_a01'
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    command='exec '+shlex.join(['env','CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','MASTER_ADDR=127.0.0.1',f'MASTER_PORT={port}','/usr/bin/python3',NATIVE])+' >'+shlex.quote(str(base/'native.log'))+' 2>&1'
    if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Native session already exists')
    subprocess.run(['tmux','new-session','-d','-s',session,command],check=True)
    subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
    samples=[]
    for _ in range(2):
        time.sleep(30);state=inspect()
        if len(state['workers'])!=8 or any(len(g['pids'])!=1 for g in state['gpus']):raise ValueError('Native GPU mapping incomplete')
        if any(not approved_launcher(process(pid)['ppid'],{NATIVE:NATIVE_SHA}) for pid in state['workers']):raise ValueError('Native owner mismatch')
        parents={process(pid)['ppid'] for pid in state['workers']}
        pane=int(subprocess.check_output(['tmux','display-message','-p','-t',session,'#{pane_pid}'],text=True))
        if parents!={pane}:raise ValueError('Native launcher is not the owned tmux pane')
        samples.append(state)
    write_json(base/'native-verified.json',dict(samples=samples,script_sha256=NATIVE_SHA,note='Native script has no progress counters; ownership verified at two snapshots, collective throughput not measured.'))

def report(base, backup=None):
    rows=[]
    for name in ['D-s0','D-s1','R-s0','R-s1','W-s0','W-s1']:
        r=json.loads((base/'eval'/name/'report.json').read_text());check_train(base/'runs'/name);rows.append(r)
        arrays=np.load(base/'eval'/name/'test_costs.npz');cost=arrays['costs'];ch=arrays['choices']
        replay={'canonical':cost[:,0]}
        if cost.shape[1]==8:replay.update(uniform=cost.mean(1),constant=cost[:,r['constant']],learned=cost[np.arange(len(cost)),ch],oracle=cost.min(1))
        for key,value in replay.items():
            np.testing.assert_allclose(arrays[key],value,rtol=0,atol=0)
            if abs(float(value.mean())-r['metrics'][key])>1e-7:raise ValueError('Report replay mismatch')
        complete=json.loads((base/'eval'/name/'complete.json').read_text())
        for file,digest in complete['files'].items():
            if sha(base/'eval'/name/file)!=digest:raise ValueError('Evaluation artifact changed')
    comparisons={}
    for seed in (0,1):
        arrays={arm:np.load(base/f'eval/{arm}-s{seed}/test_costs.npz') for arm in ('D','R','W')}
        for a in arrays.values():
            np.testing.assert_array_equal(a['ids'],arrays['D']['ids']);np.testing.assert_array_equal(a['clusters'],arrays['D']['clusters'])
        comparisons[f'W_learned_minus_D_s{seed}']=paired_ci(arrays['W']['learned']-arrays['D']['canonical'],arrays['D']['clusters'])
        comparisons[f'W_learned_minus_R_learned_s{seed}']=paired_ci(arrays['W']['learned']-arrays['R']['learned'],arrays['D']['clusters'])
    write_json(base/'REPORT.json',dict(rows=rows,comparisons=comparisons,mean_over_seeds={arm:{k:float(np.mean([r['metrics'][k] for r in rows if r['arm']==arm])) for k in next(r['metrics'] for r in rows if r['arm']==arm)} for arm in ('D','R','W')},flags=['conditional_suffix_only','two_seeds','unequal_block_compute','weights_fullstate_node_only']))
    lines=['# Warm-order screening results','','Conditional suffix NLL, 32 observed prefix / 480 scored tokens. Token budgets match; total training compute does not. Oracle is diagnostic only.','','| Arm | Seed | Canonical | Uniform | Constant | Learned | Oracle |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+r['arm']+' | '+str(r['seed'])+' | '+' | '.join(f"{r['metrics'][k]:.5f}" if k in r['metrics'] else '—' for k in ('canonical','uniform','constant','learned','oracle'))+' |')
    lines+=['','Paired document-cluster intervals and compute counts are in REPORT.json and train complete receipts. No claim of unconditional LM quality or equal-compute superiority.']
    (base/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files=[base/'REPORT.json',base/'REPORT.md',base/'gpu-probe.json',base/'execution/source.json',base/'inputs/complete.json']
    for folder in [base/'eval',base/'runs']:
        files.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.npz') and p.name!='fullstate.pt')
    files.extend((base/'eval').rglob('router.pt'))
    with tarfile.open(base/'tier1.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
    if (base/'tier1.tar.gz').stat().st_size>25_000_000:raise ValueError('Tier1 too large; no export')
    backup=Path(backup) if backup is not None else Path('/mnt/local/_backup/deep-llms_th2/warm_order_20261118_a01')
    backup.mkdir(parents=True,exist_ok=False);large={}
    for name in ['D-s0','D-s1','R-s0','R-s1','W-s0','W-s1']:
        (backup/name).mkdir()
        for file in ('weights.pt','fullstate.pt','complete.json','recipe.json','checkpoint.json'):
            src=base/'runs'/name/file;dst=backup/name/file;shutil.copyfile(src,dst)
            digest=sha(src)
            if sha(dst)!=digest:raise ValueError('Node backup hash mismatch')
            large[str(dst)]=digest
    shutil.copyfile(base/'tier1.tar.gz',backup/'tier1.tar.gz')
    if sha(backup/'tier1.tar.gz')!=sha(base/'tier1.tar.gz'):raise ValueError('Small backup mismatch')
    write_json(base/'backup.json',dict(node_checkpoint_copies=large,archive_sha256=sha(base/'tier1.tar.gz'),bytes=(base/'tier1.tar.gz').stat().st_size,files={str(p.relative_to(base)):sha(p) for p in files},large_artifacts='weights/fullstate retained in runs on node'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);a=ap.parse_args();base=Path(a.base)
    if socket.gethostname()!='thiennh-p6-ohmv-worker-0':raise ValueError('Wrong node')
    check_source(base);expected=json.loads((base/'execution/admission.json').read_text())['ownership'];current=inspect()
    if ownership(current['gpus'])!=ownership(expected['gpus']):raise ValueError('GPU ownership changed before reclaim')
    for pid,r in expected['processes'].items():
        if process(int(pid))!=r:raise ValueError('Process identity changed')
    guard=Path('/mnt/local/_gpu_guard/DISABLED')
    if sha(guard.parent/'gpu_guard.sh')!=GUARD_HASH:raise ValueError('Guard protocol changed')
    marker=json.dumps(dict(owner_pid=os.getpid(),base=str(base),utc=time.time()))
    with guard.open('x') as f:f.write(marker)
    inode=guard.stat().st_ino
    time.sleep(30)  # Let any preexisting reconcile pass finish while native remains busy.
    current=inspect()
    if ownership(current['gpus'])!=ownership(expected['gpus']) or any(process(int(pid))!=r for pid,r in expected['processes'].items()):raise ValueError('Ownership changed during guard hold; no signals')
    enable_subreaper();stop_burns(base/'execution',{NATIVE:NATIVE_SHA})
    def interrupted(signum,frame):raise KeyboardInterrupt('Owned queue interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    active=[];failures=[]
    try:
        check_source(base)
        p=spawn(base,'probe','R-s0',0,[sys.executable,'-u','-m','efficient_llm.warm_order.probe','--base',str(base)])
        if p['child'].wait()!=0:raise ValueError('Numerical or real-entrypoint probe failed; inspect log')
        # Same full schedule and state: the measured40 steps are continued, not repeated.
        for gpu,name in enumerate(['D-s0','D-s1','R-s0','R-s1','W-s0','W-s1']):
            argv=[sys.executable,'-u','-m','efficient_llm.warm_order.train','--config',str(base/f'inputs/{name}.json'),'--out',str(base/f'runs/{name}')]
            if name=='R-s0':argv.append('--resume')
            active.append(spawn(base,'train',name,gpu,argv))
        while active:
            write_json(base/'status.json',dict(active=[{k:v for k,v in x.items() if k!='child'} for x in active],failures=failures,utc=time.time()))
            for job in list(active):
                code=job['child'].poll()
                if code is None:continue
                active.remove(job);name=job['name']
                if code!=0:failures.append(dict(name=name,stage=job['stage'],code=code));continue
                if job['stage']=='train':
                    check_train(base/f'runs/{name}');seed=int(name[-1]);plan=json.loads(Path('resources/warm_order_data_20261118.json').read_text())
                    indices=str(base/f'inputs/eval-s{seed}.npz');hashes={**plan['files'],indices:sha(indices)}
                    cfg=dict(run=str(base/f'runs/{name}'),**plan['stems'],indices=indices,input_hashes=hashes)
                    write_json(base/f'eval-config-{name}.json',cfg)
                    active.append(spawn(base,'eval',name,job['gpu'],[sys.executable,'-u','-m','efficient_llm.warm_order.evaluate','--config',str(base/f'eval-config-{name}.json'),'--out',str(base/f'eval/{name}')]))
            if active:time.sleep(10)
        if failures:raise ValueError('Some owned project jobs failed: '+repr(failures))
        report(base);write_json(base/'scientific-complete.json',dict(ready=True,utc=time.time(),backup_sha256=sha(base/'backup.json')))
    except BaseException as e:
        write_json(base/'failure.json',dict(error=repr(e),utc=time.time()));traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        clean_owned_children();time.sleep(30);require_free(list(range(8)));native(base)
        if guard.exists() and guard.stat().st_ino==inode and guard.read_text()==marker:guard.unlink()
        else:raise ValueError('Guard hold owner changed; refusing removal')
        write_json(base/'terminal.json',dict(success=(base/'scientific-complete.json').exists(),native_verified=True,utc=time.time()))
if __name__=='__main__':main()
