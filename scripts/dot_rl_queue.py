"""Owned eight-lane queue; each P-rand lane immediately continues to its two FR fits."""
import argparse,dataclasses,json,os,subprocess,sys,tarfile,threading,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from efficient_llm.dot_pool.rl import *

class Queue:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False);self.lock=threading.RLock();self.start=time.time();self.flags=['model weights/fullstate not yet off-node','RL arms use OOD-RL composition rewards; controls do not','SDPA grouped whole-question dispatch; speed measured, not assumed1.2x','N uses predeclared J-RL seed0 first200 speed; both seeds retained']
        self.runs=[];self.jobs=[];self.source=Path.cwd().parent/'source-manifest.json';self.data=self.root.parent/'data';self.cfgdir=self.root/'configs';self.cfgdir.mkdir();self.j0_failed=False
        self.manifest=json.loads(self.source.read_text());self.mapping=[(arm,seed,gpu) for gpu,(arm,seed) in enumerate((a,s) for a in ARMS for s in [0,1])]
    def status(self,phase,**kw):
        with self.lock:write(self.root/'status.json',dict(phase=phase,utc=time.time(),elapsed=time.time()-self.start,flags=self.flags,**kw))
    def command(self,name,args,gpu):
        for rel,expected in self.manifest['files'].items():assert sha(Path.cwd()/rel)==expected,rel
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu) if gpu is not None else '', 'OMP_NUM_THREADS':'2'}
        for key in ['RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_PORT']:env.pop(key,None)
        start=time.time()
        with (self.root/(name+'.log')).open('x') as f:
            p=subprocess.run([sys.executable,'-u','-m',*args],env=env,stdout=f,stderr=subprocess.STDOUT)
        with self.lock:
            self.jobs.append(dict(name=name,gpu=gpu,start=start,end=time.time(),returncode=p.returncode))
            if p.returncode:self.flags.append(dict(job=name,flag='project command failed; preserve logs, other lanes continue',returncode=p.returncode))
        return p.returncode==0
    def config(self,arm,seed,gpu,**kw):
        cfg=dict(arm=arm,seed=seed,gpu=gpu,model=dataclasses.asdict(Config(stem=0,coda=0,arm='L',backend='sdpa')),batch=4096,lr=.001,data=str(self.data),output=str(self.root/f'{arm}-s{seed}'),data_sha256=sha(self.data/'manifest.json'),source_sha256=sha(self.source),budget_path=str(self.root/'budget.json'),spec_sha256=SPEC_SHA,**kw)
        name=f'{arm}-s{seed}';write(self.cfgdir/(name+'.json'),cfg)
        with self.lock:self.runs.append(cfg)
        return cfg
    def backup(self,run):
        small=[p for p in run.iterdir() if p.is_file() and (p.suffix in ['.json','.jsonl','.npz','.csv','.md','.png'] or p.name.endswith('_router.pt')) and p.name!='artifact_manifest.json']
        records=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),tier=1 if p in small else 2 if p.name=='final_weights.pt' else 3,off_node=False,locations=[str(p)]) for p in sorted(run.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json']
        write(run/'artifact_manifest.json',dict(artifacts=records));small.append(run/'artifact_manifest.json')
        dest=self.root/'tier1';dest.mkdir(exist_ok=True)
        archive=dest/(run.name+'.tar.gz')
        with tarfile.open(archive,'w:gz') as tar:
            for p in sorted(small):tar.add(p,arcname=p.name,recursive=False)
        write(dest/(run.name+'.json'),dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,ready_for_small_export=archive.stat().st_size<20*1024**2))
    def lane(self,triple):
        arm,seed,gpu=triple;cfg=self.config(arm,seed,gpu);name=f'{arm}-s{seed}'
        ok=self.command(name,['efficient_llm.dot_pool.rl_run','--config',str(self.cfgdir/(name+'.json'))],gpu)
        if arm=='J-RL' and seed==0 and not ok:self.j0_failed=True
        if ok and (Path(cfg['output'])/'complete.json').exists():self.backup(Path(cfg['output']))
        if arm=='P-rand' and ok:
            for fr in ['FR-adaptive','FR-fixed']:
                fc=self.config(fr,seed,gpu,parent=cfg['output']);name=f'{fr}-s{seed}'
                ok=self.command(name,['efficient_llm.dot_pool.rl_run','--config',str(self.cfgdir/(name+'.json'))],gpu)
                if ok and (Path(fc['output'])/'complete.json').exists():self.backup(Path(fc['output']))
    def monitor_budget(self,stop):
        while not stop.wait(2):
            if self.j0_failed and not (self.root/'budget.json').exists():
                f=self.root/'J-RL-s1/first200.json'
                if f.exists():
                    r=json.loads(f.read_text());b=choose_budget(r['seconds_per_epoch']);b['reference']='J-RL seed1 fallback because seed0 failed';write(self.root/'budget.json',b)
                    with self.lock:self.flags.append('Budget source fallback: J-RL seed0 failed')
    def run(self):
        self.status('phase0_tests')
        if not self.data.exists():make_data(self.data)
        expected=json.loads((Path.cwd()/'resources/dot_rl_data_expected.json').read_text())
        actual=json.loads((self.data/'manifest.json').read_text());assert expected==actual,'Local/node data mismatch'
        def test(pair):
            arm,gpu=pair;out=self.root/('checks-'+arm)
            ok=self.command('checks-'+arm,['scripts.dot_rl_checks','--arm',arm,'--data',str(self.data),'--output',str(out)],gpu)
            f=out/'checks.json';r=json.loads(f.read_text()) if f.exists() else dict(passed=False,flag='missing test receipt')
            if not r['passed']:self.flags.append(dict(arm=arm,flag='production test failed; suspect, inspect tests; queue continues per spec',tests=r))
            return r
        with ThreadPoolExecutor(max_workers=3) as pool:checks=list(pool.map(test,[('L',0),('P-rand',2),('J-RL',4)]))
        write(self.root/'tests_summary.json',dict(checks=checks,elapsed=time.time()-self.start))
        if time.time()-self.start>1800:self.flags.append('Phase0 exceeded0.5h')
        self.status('phase1_main_with_FR_lanes',mapping=self.mapping);start_main=time.time()
        stop=threading.Event();monitor=threading.Thread(target=self.monitor_budget,args=(stop,));monitor.start()
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures=[pool.submit(self.lane,t) for t in self.mapping]
                for future in futures:
                    try:future.result()
                    except Exception:self.flags.append(dict(flag='lane exception',trace=traceback.format_exc()))
        finally:stop.set();monitor.join()
        if time.time()-self.start>12*3600:self.flags.append('Total elapsed exceeded12h target')
        if (self.root/'budget.json').exists() and json.loads((self.root/'budget.json').read_text())['minimum_overrun']:self.flags.append('2000epoch minimum exceeds10.5h phase1 target')
        write(self.root/'run_index.json',dict(runs=self.runs,flags=self.flags,jobs=self.jobs,start_utc=self.start,main_start=start_main,elapsed=time.time()-self.start))
        self.status('phase3_report')
        ok=self.command('report',['scripts.dot_rl_report','--root',str(self.root)],None)
        receipt=self.root/'report_complete.json'
        if not ok or not receipt.exists() or not json.loads(receipt.read_text())['reported']:raise RuntimeError('Incomplete report; preserved evidence, no false completion')
        self.status('complete',reported=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--receipt',required=True);a=ap.parse_args();q=Queue(a.root)
    try:q.run();write(a.receipt,dict(reported=True,root=a.root))
    except Exception:q.status('project_failure',trace=traceback.format_exc());raise
