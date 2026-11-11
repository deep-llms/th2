"""Registered automatic Phase0→E1→E2→evaluation/report queue inside owned supervisor."""
import argparse,dataclasses,json,os,subprocess,sys,time,traceback,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from efficient_llm.dot_pool import SPEC_SHA
from efficient_llm.dot_pool.model import Config
from efficient_llm.dot_pool.data import write,sha

ARMS={'E1':['D','L','SO','DOT','SR'],'E2':['D','PL','SO','DOT','SR']}
class Queue:
 def __init__(self,root,python):
  self.lock=threading.RLock();self.root=Path(root);self.python=python;self.flags=[];self.start=time.time();self.runs=[];self.root.mkdir(parents=True,exist_ok=False)
  self.source_manifest=Path.cwd().parent/'source-manifest.json';self.data=self.root.parent/'data';self.ready=json.loads((self.data/'ready.json').read_text());self.cfgdir=self.root/'configs';self.cfgdir.mkdir()
  if sha(self.data/'e2/manifest.json')!=self.ready['data_manifest_sha256']:raise ValueError('Data manifest changed')
 def status(self,phase,**detail):
  with self.lock:
   value=dict(phase=phase,elapsed_hours=(time.time()-self.start)/3600,flags=list(self.flags),**detail);write(self.root/'status.json',value)
   with (self.root/'phases.jsonl').open('a') as f:f.write(json.dumps(dict(value,utc=time.time(),spec_sha256=SPEC_SHA))+'\n')
 def command(self,name,argv,gpus=None):
  if self.source_manifest.exists():
   for rel,expected in json.loads(self.source_manifest.read_text())['files'].items():
    if sha(Path.cwd()/rel)!=expected:raise RuntimeError('Frozen source modified: '+rel)
  env={**os.environ,'OMP_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','NCCL_NVLS_ENABLE':'0'}
  for key in ['RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_PORT']:env.pop(key,None)
  if gpus is not None:env['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))
  with (self.root/(name+'.log')).open('x') as log:p=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
  if p.returncode:self.flags.append(dict(job=name,returncode=p.returncode,log=str(self.root/(name+'.log'))));self.status('project_failure_recorded',job=name)
  return p.returncode==0
 def config(self,study,arm,seed,name,steps,lr,benchmark=False,**extra):
  mc=Config.production(study,arm,backend='fa4',expert_kernel='grouped')
  cfg=dict(study=study,arm=arm,seed=seed,name=name,steps=int(steps),lr=lr,batch=4096 if study=='E1' else 512,microbatch=4096 if study=='E1' else 64,eval_batch=32,model=dataclasses.asdict(mc),data=str(self.data/study.lower()),output=str(self.root/name),benchmark=benchmark,spec_sha256=SPEC_SHA,data_ready_sha256=sha(self.data/'ready.json'),source_manifest_sha256=sha(self.source_manifest),**extra)
  write(self.cfgdir/(name+'.json'),cfg);return cfg
 def train(self,cfg,gpus):
  path=self.cfgdir/(cfg['name']+'.json');argv=[self.python]
  if len(gpus)>1:argv+=['-m','torch.distributed.run','--standalone','--nproc_per_node',str(len(gpus)),'-m','efficient_llm.dot_pool.train']
  else:argv+=['-m','efficient_llm.dot_pool.train']
  argv+=['--config',str(path)];ok=self.command(cfg['name'],argv,gpus)
  terminal=Path(cfg['output'])/'train_complete.json';ok=ok and terminal.exists()
  if ok and not cfg['benchmark']:
   self.command(cfg['name']+'-routes',[self.python,'-m','scripts.dot_pool_evaluate','routes','--run',cfg['output']],gpus[:1])
   self.backup(Path(cfg['output']))
  return json.loads(terminal.read_text()) if ok else None
 def parallel(self,jobs):
  with ThreadPoolExecutor(max_workers=10) as p:return list(p.map(lambda a:self.train(*a),jobs))
 def backup(self,path):
  # Publish an immutable compact bundle per completed run; watcher pulls exact files.
  import tarfile
  small=[p for p in path.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.npz','.csv','.md','.png') and p.name not in ('downstream_scores.json','downstream_raw.json','artifact_manifest.json')]
  manifest=[]
  for p in sorted(path.rglob('*')):
   if p.is_file() and p.name!='artifact_manifest.json':manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),tier=1 if p in small else 3 if p.suffix=='.pt' and 'pre_decay' in p.parts else 2 if p.name=='final_weights.pt' else 'local_training_state',locations=[str(p)],verified_utc=time.time(),off_node=False))
  write(path/'artifact_manifest.json',dict(artifacts=manifest,flags=['Tier2/3 not yet off-node; awaiting approved large-transfer mechanism']))
  small.append(path/'artifact_manifest.json');dest=self.root/'tier1';dest.mkdir(exist_ok=True)
  revision='analysis' if (path/'analysis_complete.json').exists() else 'training';archive=dest/(path.name+'-'+revision+'.tar.gz')
  with tarfile.open(archive,'w:gz') as tar:
   for f in small:tar.add(f,arcname=f.name,recursive=False)
  if archive.stat().st_size>20*1024*1024:self.flags.append(dict(job=path.name,flag='Tier1 bundle exceeds small-transfer cap; export compact report separately'))
  write(dest/(path.name+'-'+revision+'.json'),dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),ready_for_small_export=archive.stat().st_size<=20*1024*1024))
  with self.lock:write(self.root/'manifest.json',dict(artifacts=[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(dest.iterdir()) if f.suffix=='.json'],tier2_3='not yet off-node'))
 def run(self):
  self.status('phase0_tests')
  request_args=['/mnt/local/conda-py311/envs/eval/bin/python3.11','-m','eval.a2_requests','prepare','--tokenizer',self.ready['tokenizer'],'--dataset-root','/mnt/local/_data/deep-llms_th2/benchmarks_zero_20261023','--manifest','resources/downstream_english_20261023.json','--tasks','hellaswag','piqa','arc_easy','lambada_openai','--max-length','512','--output',str(self.root/'requests.json')]
  self.command('requests',request_args,[])
  self.command('gpu-tests',[self.python,'-m','scripts.dot_pool_checks','--device','cuda:0','--output',str(self.root/'tests.json')],[0])
  # Native kernel checks have exact float32 reference and grouped gradient checks.
  self.command('kernel-tests',[self.python,'-m','scripts.dot_pool_kernel_checks','--output',str(self.root/'kernel_tests.json')],[0])
  k=json.loads((self.root/'kernel_tests.json').read_text()) if (self.root/'kernel_tests.json').exists() else {}
  if not k.get('passed'):
   # Do not silently use an unverified numerical kernel. Independent CPU/data work survives.
   self.flags.append(dict(flag='production kernel check failed; automatic SDPA/loop fallback, timing remeasured',detail=k));self.kernel_fallback=True
  else:self.kernel_fallback=False
  tests=json.loads((self.root/'tests.json').read_text()) if (self.root/'tests.json').exists() else {'failed':['missing']}
  if tests.get('failed'):self.flags.append(dict(flag='core test failures; affected claims suspect pending review',tests=tests['failed']))
  self.status('phase0_production_overfit')
  pairs=[(study,arm) for study,arms in ARMS.items() for arm in arms]
  def fit_lane(pair):
   gpu,items=pair
   for study,arm in items:
    name='overfit-'+study+'-'+arm;out=self.root/(name+'.json')
    argv=[self.python,'-m','scripts.dot_pool_overfit','--study',study,'--arm',arm,'--data',str(self.data),'--output',str(out)]
    if self.kernel_fallback:argv+=['--fallback']
    self.command(name,argv,[gpu])
    if not out.exists() or not json.loads(out.read_text())['passed']:self.flags.append(dict(flag='T7 production overfit failed after two LR attempts; arm suspect',study=study,arm=arm))
  with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(fit_lane,[(g,pairs[g::8]) for g in range(8)]))
  self.status('phase0_e1_lr')
  lrs=[3e-4,1e-3];lrjobs=[(self.config('E1','D',0,f'cal-e1-lr{i}',500*77,lr),[i]) for i,lr in enumerate(lrs)]
  self.parallel(lrjobs);valid=[]
  for cfg,_ in lrjobs:
   f=Path(cfg['output'])/'final_eval.json'
   if f.exists():valid.append((json.loads(f.read_text())['metrics']['id_dev']['accuracy'],cfg['lr']))
  lr1=sorted(valid,key=lambda x:(-x[0],x[1]))[0][1] if valid else 3e-4
  if not valid:self.flags.append(dict(flag='E1 LR calibration failed; provisional lower LR used'))
  self.status('phase0_e1_throughput')
  mapping=[('SO',0,0),('DOT',0,1),('SR',0,2),('SO',1,3),('DOT',1,4),('SR',1,5),('D',0,6),('L',0,6),('D',1,7),('L',1,7)]
  bench1=[(self.config('E1',a,s,f'bench-e1-{a}-s{s}',2000,lr1,True),[g]) for a,s,g in mapping];results=self.parallel(bench1);times=[r['seconds_per_update'] for r in results if r]
  speed=max(times) if times else 1.;epochs=max(2000,min(10000,int(10800/speed/77/500)*500))
  if epochs*77*speed>10800:self.flags.append(dict(flag='E1 minimum exceeds target3h',predicted_seconds=epochs*77*speed))
  self.status('phase0_e2_throughput')
  speed2={}
  for arm in ARMS['E2']:
   cfg=self.config('E2',arm,0,'bench-e2-'+arm,120,.001,True);r=self.train(cfg,list(range(8)))
   if r:
    rows=[json.loads(l) for l in (Path(cfg['output'])/'metrics.jsonl').read_text().splitlines()];a=next(x for x in rows if x['step']==20);b=rows[-1];speed2[arm]=(b['elapsed']-a['elapsed'])/(b['step']-a['step'])
   else:speed2[arm]=3.;self.flags.append(dict(flag='throughput missing; provisional3s/update',arm=arm))
  self.status('phase0_e2_lr');lr2={}
  for arm in ['D','SO']:
   options=[]
   for lr in [.001,.002,.004]:
    cfg=self.config('E2',arm,0,f'cal-e2-{arm}-{lr}',int(.25e9//262144),lr);r=self.train(cfg,list(range(8)))
    f=Path(cfg['output'])/'final_eval.json'
    if r and f.exists():options.append((lr,json.loads(f.read_text())['metrics']['val']['nll']))
   if options:
    best=min(v for _,v in options);lr2[arm]=min(lr for lr,v in options if v<=best+.005)
   else:lr2[arm]=.001;self.flags.append(dict(flag='LR calibration missing; provisional lower LR',arm=arm))
  choices=[1.,1.25,1.5,2.];fits=[t for t in choices if sum(speed2.values())*2*int(t*1e9//262144)*1.05<=21600];tokens=max(fits) if fits else 1.
  if not fits:self.flags.append(dict(flag='E2 minimum exceeds6h target'))
  write(self.root/'budget.json',dict(E1_epochs=epochs,E2_billions=tokens,E1_lr=lr1,E2_lr=lr2,E1_slowest_seconds=speed,E2_seconds=speed2,flags=self.flags))
  self.status('phase1_e1',epochs=epochs)
  jobs=[(self.config('E1',a,s,f'E1-{a}-s{s}',epochs*77,lr1),[g]) for a,s,g in mapping];self.parallel(jobs);self.runs.extend(cfg for cfg,_ in jobs)
  self.status('phase2_e2',tokens_billions=tokens)
  for seed in [0,1]:
   for arm in ARMS['E2']:
    cfg=self.config('E2',arm,seed,f'E2-{arm}-s{seed}',int(tokens*1e9//262144),lr2['D' if arm=='D' else 'SO']);self.status('phase2_e2',current=cfg['name']);self.train(cfg,list(range(8)));self.runs.append(cfg)
  self.status('phase3_evaluation')
  completed=[c for c in self.runs if (Path(c['output'])/'final_eval.json').exists()]
  language=[c for c in completed if c['study']=='E2']
  def downstream_lane(pair):
   gpu,configs=pair
   for cfg in configs:self.command(cfg['name']+'-downstream',[self.python,'-m','scripts.dot_pool_evaluate','downstream','--run',cfg['output']],[gpu])
  with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(downstream_lane,[(g,language[g::8]) for g in range(8)]))
  # Lane scheduling ensures at most one analysis/eval worker per GPU.
  def lane(pair):
   gpu,configs=pair
   for cfg in configs:self.command(cfg['name']+'-analysis',[self.python,'-m','scripts.dot_pool_evaluate','diagnostics','--run',cfg['output']],[gpu]);self.backup(Path(cfg['output']))
  with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(lane,[(g,completed[g::8]) for g in range(8)]))
  self.status('phase3_report')
  self.command('report',[self.python,'-m','scripts.dot_pool_report','--root',str(self.root)],[])
  # Conditional constant-LR extension, original endpoints and states stay immutable.
  elapsed=time.time()-self.start;trends=[]
  for seed in [0,1]:
   path=self.root/f'E1-L-s{seed}';es=sorted([json.loads(p.read_text()) for p in path.glob('eval-*.json')],key=lambda x:x['step'])
   if es:
    end=es[-1];before=[e for e in es if e['step']<=end['step']-500*77]
    if before:trends.append(end['metrics']['ood_dev']['accuracy']-before[-1]['metrics']['ood_dev']['accuracy'])
  seconds_per_epoch=speed*77
  extra=max(0,int((43200-elapsed-600)/seconds_per_epoch/50)*50) if elapsed<41400 and len(trends)==2 and sum(trends)>0 else 0
  eligible=[c for c in self.runs if c['study']=='E1' and (Path(c['output'])/'final_state/complete.json').exists()]
  if len(eligible)!=10:extra=0
  write(self.root/'phase4.json',dict(elapsed_seconds=elapsed,loop_last500_deltas=trends,extra_epochs=extra,reason='under11.5h and mean loop OOD-dev rising; reserve10min for final eval' if extra else 'condition not met or no safe complete ten-run extension'))
  if extra:
   self.status('phase4_extension',extra_epochs=extra);jobs=[]
   for arm,seed,gpu in mapping:
    old=self.root/f'E1-{arm}-s{seed}';cfg=self.config('E1',arm,seed,f'E1-{arm}-s{seed}-extended',(epochs+extra)*77,lr1,resume=str(old/'final_state'));jobs.append((cfg,[gpu]))
   self.parallel(jobs)
   with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(lane,[(g,[c for c,_ in jobs][g::8]) for g in range(8)]))
   self.command('report-extended',[self.python,'-m','scripts.dot_pool_report','--root',str(self.root)],[])
  self.status('final_report_backup')
  import tarfile
  files=[p for p in self.root.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.md','.png') and p.name!='requests.json']
  with tarfile.open(self.root/'REPORT_SMALL.tar.gz','w:gz') as tar:
   for f in files:tar.add(f,arcname=f.name,recursive=False)
  write(self.root/'REPORT_SMALL.json',dict(path=str(self.root/'REPORT_SMALL.tar.gz'),sha256=sha(self.root/'REPORT_SMALL.tar.gz'),bytes=(self.root/'REPORT_SMALL.tar.gz').stat().st_size))
  write(self.root/'queue_complete.json',dict(completed_runs=len(completed),expected_runs=20,flags=self.flags,elapsed_hours=(time.time()-self.start)/3600))
  self.status('complete_with_flags' if self.flags else 'complete',completed_runs=len(completed))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--receipt');a=p.parse_args();q=Queue(a.root,sys.executable)
 original=q.config
 def config(*args,**kwargs):
  cfg=original(*args,**kwargs)
  if getattr(q,'kernel_fallback',False):cfg['model'].update(backend='sdpa',expert_kernel='loop');write(q.cfgdir/(cfg['name']+'.json'),cfg)
  return cfg
 q.config=config
 try:
  q.run()
  if not (q.root/'report_complete.json').exists():raise RuntimeError('Report missing')
  if a.receipt:write(a.receipt,dict(reported=True,root=str(q.root),queue_sha256=sha(q.root/'queue_complete.json'),report_sha256=sha(q.root/'REPORT.md')))
 except BaseException as e:q.flags.append(dict(flag='coordinator_exception',error=repr(e),trace=traceback.format_exc()));q.status('failed');raise
if __name__=='__main__':main()
