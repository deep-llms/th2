"""Registered F10 ordering, eight owned lanes, immutable inputs, retained flags."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from eval.a2_family_f10 import write
from eval.a2_family_f9 import read
from eval.a2_family_f9_v110 import export_items
from scripts.a2_family_f9_run import locked
from scripts.a2_quant_io import lock
from efficient_llm.diagnostic_data import sha256


class Queue:
    def __init__(self,cfg):
        self.cfg=cfg;self.root=Path(cfg['output']);self.root.mkdir(parents=True,exist_ok=False)
        self.source=Path(cfg['source_manifest']['path']).parent/'source'
        self.flags=list(cfg['retained_flags'])+[dict(test='Winogrande_conf_semantics',detail='Canonical F9 task scores suffix conditional on candidate choice; F10 uses gold-conditioned suffix LL/character as a qualified proxy. MC8/MC7 confidence inherit this qualification; no extra model run.')];self.mutex=threading.Lock();self.start=time.monotonic()
        self.environment={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                          'HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline','OMP_NUM_THREADS':'1',
                          'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
                          'PYTHONPATH':str(self.source),'A2_CODE_COMMIT':cfg['code_commit']}
        self.verify();self.batch=1;self.generation_reuse={}
        write(self.root/'flags.json',dict(flags=self.flags))
        for entry in cfg['oneshot_entries']+sum(cfg['groups'],[]):write(self.root/'entries'/(entry['model_id']+'.json'),entry)

    def verify(self):
        for name,h in locked(self.cfg['source_manifest'])['files'].items():
            if sha256(self.source/name)!=h:raise ValueError('Frozen source changed: '+name)
        for key in ('prepared','FT16','input_manifest','f9_manifest','f10a'):locked(self.cfg[key])

    def flag(self,test,error):
        with self.mutex:
            self.flags.append(dict(test=test,detail=str(error)));write(self.root/'flags.json',dict(flags=self.flags))
        print('F10_FLAG',test,str(error)[:300],flush=True)

    def progress(self,phase):
        write(self.root/'progress.json',dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),
              elapsed_seconds=time.monotonic()-self.start,flags=self.flags,
              completed=len(list(self.root.glob('*/*/complete.json')))))
        print('F10_PHASE',phase,flush=True)

    def command(self,args,log,gpu=None):
        log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('x') as stream:
            subprocess.run(args,cwd=self.source,env={**self.environment,'CUDA_VISIBLE_DEVICES':'' if gpu is None else str(gpu)},
                           stdout=stream,stderr=subprocess.STDOUT,check=True)

    def worker(self,entry,gpu,mode,label,batch=None):
        name=entry['model_id'];out=self.root/label/name
        common=self.cfg['benchmarks']['smollm3' if entry['family']=='smollm3' else 'gpt_neox']
        prepared=common['prepared'] if mode=='nine' else self.cfg['prepared']
        args=[sys.executable,'-u','-m','eval.a2_family_f10_score','--entry',str(self.root/'entries'/(name+'.json')),
              '--output',str(out),'--mode',mode,'--prepared',prepared['path'],'--tokenizer',common['tokenizer'],
              '--batch-size',str(self.cfg['batches'][entry['batch_family']]),
              '--generation-batch-size',str(batch or self.batch)]
        if mode=='new' and name in self.generation_reuse:args+=['--generation-reuse',self.generation_reuse[name]]
        self.command(args,out.parent/(name+'.log'),gpu)
        if not read(out/'complete.json')['execution_complete']:raise ValueError('Missing worker completion')
        if mode=='ft14':
            value=read(out/'FT14.json')
            if not value['passed']:self.flag('FT14:'+name,value)
        if mode=='nine':
            args=[common['eval_python'],'-u','-m','eval.a2_requests','replay',
                  '--tokenizer',common['tokenizer'],'--input-spec',common['input_spec']['path'],
                  '--dataset-root',common['dataset_root'],'--manifest',common['manifest']['path'],
                  '--tasks',*common['tasks'],'--prepared',common['prepared']['path'],
                  '--scored',str(out/'scores.json'),'--output',str(out/'replay.json')]
            if entry['family']=='smollm3':args+=['--tokenizer-family','smollm3','--tokenizer-manifest',common['tokenizer_manifest']['path']]
            self.command(args,out/'replay.log')
            exported,_=export_items(read(out/'replay.json'),locked(prepared),read(out/'scores.json'))
            write(out/'per_item.json',dict(exported,model=entry))
        if mode=='new' and not entry.get('oneshot') and entry['role']=='parent':
            value=read(out/'generation_items.json')
            if value['exact_match']<.05:self.flag('TriviaQA_parent_below_5pct:'+name,value['exact_match'])
        print('F10_WORKER_COMPLETE',label,name,flush=True)

    def lanes(self,entries,mode,label):
        def lane(gpu):
            for entry in entries[gpu::8]:
                try:self.worker(entry,gpu,mode,label)
                except Exception as error:self.flag(label+':'+entry['model_id'],repr(error))
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(lane,range(8)))

    def run(self):
        self.progress('FT14');self.lanes(self.cfg['oneshot_entries'],'ft14','FT14')
        self.progress('FT15');parent=next(e for e in self.cfg['groups'][0] if e['arm']=='P300' and e['dataset']=='dclm')
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(self.worker,parent,b-1,'generation','FT15_batch'+str(b),b) for b in (1,2)]
                for future in futures:future.result()
            a=read(self.root/'FT15_batch1'/parent['model_id']/'generation_items.json')
            b=read(self.root/'FT15_batch2'/parent['model_id']/'generation_items.json')
            if a['task_identity']!=b['task_identity'] or len(a['items'])!=len(b['items']):raise ValueError('FT15 item mismatch')
            difference=100*(b['exact_match']-a['exact_match']);self.batch=2 if abs(difference)<=.2 else 1
            write(self.root/'FT15.json',dict(passed=self.batch==2,batch_size=self.batch,difference_pp=difference,
                  batch1=lock(self.root/'FT15_batch1'/parent['model_id']/'generation_items.json'),
                  batch2=lock(self.root/'FT15_batch2'/parent['model_id']/'generation_items.json')))
            self.generation_reuse[parent['model_id']]=str(self.root/('FT15_batch'+str(self.batch))/parent['model_id']/'generation_items.json')
            if self.batch==1:self.flag('FT15_batch_fallback',difference)
        except Exception as error:
            self.flag('FT15_failure_use_batch1',repr(error));self.batch=1
            write(self.root/'FT15.json',dict(passed=False,batch_size=1,error=repr(error)))
        self.progress('F10b_nine_task_oneshots');self.lanes(self.cfg['oneshot_entries'],'nine','nine')
        for i,entries in enumerate(self.cfg['groups']):
            self.progress('F10c_group_'+str(i));self.lanes(entries,'new','new')
        self.verify();self.progress('analysis')
        from eval.a2_family_f10_report import report
        report_ok=False
        try:
            report(self.cfg,self.root);report_ok=read(self.root/'report_complete.json')['complete']
            if not report_ok:self.flag('F10_incomplete_coverage','Partial report retained; expected 11/42/34')
        except Exception as error:self.flag('F10_report',repr(error))
        self.progress('backup')
        from scripts.a2_family_f10_backup import backup
        proof=backup(self.root,self.cfg)
        write(self.root/'verification.json',dict(backup=proof,report_complete=report_ok,
              counts=dict(nine=len(list((self.root/'nine').glob('*/per_item.json'))),
                          new=len(list((self.root/'new').glob('*/likelihood_items.json'))),
                          generation=len(list((self.root/'new').glob('*/generation_items.json')))),
              FT15=read(self.root/'FT15.json'),FT14={p.parent.name:read(p)['passed'] for p in (self.root/'FT14').glob('*/FT14.json')},
              report_replay=read(self.root/'report_complete.json') if report_ok else None,flags=self.flags,
              compact_files={p.name:lock(p) for p in [self.root/'REPORT_FULL_v1.11.md',self.root/'F10_MODELS.csv',self.root/'F10a_TESTS.csv'] if p.exists()}))
        self.progress('finished')
        write(self.root/'complete.json',dict(execution_complete=True,report_complete=report_ok,
              nine_models=len(list((self.root/'nine').glob('*/per_item.json'))),
              new_models=len(list((self.root/'new').glob('*/likelihood_items.json'))),
              generation_models=len(list((self.root/'new').glob('*/generation_items.json'))),flags=self.flags,
              elapsed_seconds=time.monotonic()-self.start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--receipt',required=True)
    a=p.parse_args();cfg=read(a.config);Queue(cfg).run();write(a.receipt,read(Path(cfg['output'])/'complete.json'))
