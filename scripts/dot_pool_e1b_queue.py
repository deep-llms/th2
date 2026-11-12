"""v1.5 only: production checks, eight E1b fits, CPU readings and compact backup."""
import argparse, dataclasses, json, math, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from efficient_llm.dot_pool import SPEC_SHA, E1B_SPEC_SHA
from efficient_llm.dot_pool.data import sha, write
from scripts.dot_pool_queue import Queue
from scripts.dot_pool_e1b_checks import config as model_config
from scripts.dot_pool_e1b_analysis import analyze

ARMS=['D','L','SO','DOT']
BASE=Path('/mnt/local/_outputs/deep-llms_th2/dot_pool_20261112')

def choose_budget(records, now=None):
    if len(records)!=2 or any(r['step']!=15400 or r['start_step']!=0 or not math.isfinite(r['training_seconds']) or r['training_seconds']<=0 for r in records):
        raise ValueError('Both DOT first200epoch measurements required')
    speed=max(r['training_seconds']/200 for r in records)
    now=time.time() if now is None else now
    elapsed=[now-r['invocation_start_utc'] if 'invocation_start_utc' in r else r['training_seconds'] for r in records]
    capacities=[200+(11*3600-e)/(r['training_seconds']/200) for e,r in zip(elapsed,records)]
    n=max(2000,math.floor(min(capacities)/500)*500)
    return dict(epochs=n,seconds_per_epoch=speed,DOT_first200=records,
                minimum_overruns_11h=min(capacities)<2000,predicted_training_seconds=n*speed,
                elapsed_at_selection=elapsed,predicted_finish_from_start=[e+(n-200)*r['training_seconds']/200 for e,r in zip(elapsed,records)],
                choice='largest500 multiple fitting11h for both DOT seeds at measured first200 speed, accounting for elapsed transition; minimum2000',
                measurement='training loop including periodic dev/checkpoints; excluding final save/eval and stage transition')

class E1bQueue(Queue):
    def __init__(self,root,python):
        if SPEC_SHA!=E1B_SPEC_SHA:raise ValueError('Explicit v1.5 protocol required')
        super().__init__(root,python)
        self.flags += ['pilot SR reference has unmatched 12-layer layout and 2000epoch budget',
                       'final weights/full training state not yet off-node', 'eager execution; no optional compile speed claim']
        self.mapping=[(a,s,g) for g,(a,s) in enumerate((a,s) for a in ARMS for s in [0,1])]
    def config(self,arm,s,name,epochs,**extra):
        cfg=dict(study='E1',experiment='E1b',arm=arm,seed=s,name=name,steps=epochs*77,lr=.001,batch=4096,
                 microbatch=4096,eval_batch=32,checkpoint_epochs=500,model=dataclasses.asdict(model_config(arm,backend='fa4',expert_kernel='grouped')),
                 data=str(self.data/'e1'),output=str(self.root/name),benchmark=False,spec_sha256=SPEC_SHA,
                 data_ready_sha256=sha(self.data/'ready.json'),source_manifest_sha256=sha(self.source_manifest),**extra)
        write(self.cfgdir/(name+'.json'),cfg);return cfg
    def raw_train(self,cfg,gpu):
        started=time.time()
        ok=self.command(cfg['name'],[self.python,'-u','-m','efficient_llm.dot_pool.train','--config',str(self.cfgdir/(cfg['name']+'.json'))],[gpu])
        path=Path(cfg['output'])/'train_complete.json'
        if not ok or not path.exists():return None
        receipt=json.loads(path.read_text())
        receipt['invocation_start_utc']=started
        write(Path(cfg['output'])/'invocation.json',dict(started_utc=started,finished_utc=time.time(),gpu=gpu))
        if receipt['step']!=cfg['steps']:raise ValueError('Incomplete training endpoint')
        if receipt['flags']:self.flags.append(dict(run=cfg['name'],training_flags=receipt['flags']))
        return receipt
    def tests(self,arm,gpu):
        out=self.root/('checks-'+arm)
        self.command('checks-'+arm,[self.python,'-m','scripts.dot_pool_e1b_checks','--arm',arm,'--data',str(self.data/'e1'),'--output',str(out)],[gpu])
        ok=(out/'checks.json').exists() and json.loads((out/'checks.json').read_text())['passed']
        over=self.root/('overfit-'+arm+'.json')
        self.command('overfit-'+arm,[self.python,'-m','scripts.dot_pool_overfit','--study','E1','--arm',arm,'--data',str(self.data),'--output',str(over),'--no-stem-coda'],[gpu])
        ok=ok and over.exists() and json.loads(over.read_text())['passed']
        if not ok:self.flags.append(dict(arm=arm,flag='production test failed; arm suspect; inspect test records'))
        return ok
    def pilot(self):
        dest=self.root/'pilot_same_relation';dest.mkdir()
        for a in ['SO','DOT','SR']:
            for s in [0,1]:
                name=f'E1-{a}-s{s}'
                source=BASE/'run-a03'/name/'routes_sample.npz' if a!='SR' else BASE/'sr-recheck-a01/supervised/run'/name/'routes_sample.npz'
                analyze(source,self.data/'e1',dest/(name+'.json'))
    def finish_run(self,cfg,gpu):
        receipt=self.raw_train(cfg,gpu)
        if receipt is None:return
        run=Path(cfg['output']);ok=self.command(cfg['name']+'-routes',[self.python,'-m','scripts.dot_pool_evaluate','routes','--run',str(run)],[gpu])
        if ok and cfg['arm'] in ['SO','DOT']:
            from scripts.dot_pool_evaluate import e1_analysis
            e1_analysis(run);analyze(run/'routes_sample.npz',self.data/'e1',run/'reentry_same_relation.json')
        write(run/'analysis_complete.json',dict(complete=ok,training_only=False))
        self.backup(run)
    def run(self):
        self.status('E1b_tests',mapping=self.mapping)
        with ThreadPoolExecutor(max_workers=5) as pool:
            prior=pool.submit(self.pilot)
            checks=list(pool.map(lambda x:self.tests(*x),[(a,i*2) for i,a in enumerate(ARMS)]));prior.result()
        write(self.root/'tests_summary.json',dict(arms=ARMS,passed=checks,flags=self.flags))
        self.status('E1b_first200',mapping=self.mapping)
        warm=[(self.config(a,s,f'first200-{a}-s{s}',200),g) for a,s,g in self.mapping]
        with ThreadPoolExecutor(max_workers=8) as pool:
            receipts=list(pool.map(lambda x:self.raw_train(*x),warm))
        dots=[r for (cfg,_),r in zip(warm,receipts) if cfg['arm']=='DOT' and r]
        budget=choose_budget(dots);write(self.root/'budget.json',budget)
        if budget['minimum_overruns_11h']:self.flags.append('2000epoch minimum exceeds11h measured training target')
        jobs=[]
        for ((old,g),receipt) in zip(warm,receipts):
            if receipt is None:continue
            cfg=self.config(old['arm'],old['seed'],f"E1b-{old['arm']}-s{old['seed']}",budget['epochs'],resume=str(Path(old['output'])/'final_state'),first200_output=old['output'])
            self.runs.append(cfg);jobs.append((cfg,g))
        self.status('E1b_continuation',epochs=budget['epochs'],runs=[c['name'] for c,_ in jobs])
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(lambda x:self.finish_run(*x),jobs))
        self.status('E1b_report')
        if time.time()-self.start>12*3600:self.flags.append('Phase5 elapsed exceeds12h target')
        write(self.root/'run_index.json',dict(runs=self.runs,flags=self.flags,start_utc=self.start,elapsed=time.time()-self.start))
        if not self.command('report',[self.python,'-m','scripts.dot_pool_e1b_report','--root',str(self.root)],[]):raise RuntimeError('E1b report failed; preserve runs for CPU recovery')
        self.status('E1b_complete',reported=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
    q=E1bQueue(a.root,sys.executable)
    try:q.run();write(a.receipt,dict(reported=True,root=a.root,spec_sha256=SPEC_SHA))
    except Exception:
        q.status('project_failure',trace=traceback.format_exc());raise
