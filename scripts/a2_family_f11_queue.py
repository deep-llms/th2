"""F11b then F11a, under the unchanged native GPU supervisor."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from eval.a2_family_f11 import read,write,SPEC_SHA
from scripts.a2_quant_io import lock
from scripts.a2_family_f9_run import locked
from scripts.gpu_process import process
from scripts.gpu_status import require_free
from efficient_llm.diagnostic_data import sha256

def descriptors():
    return [dict(run_id=f'LM1-dclm-{arm}-S1-0.8B-lr1e-3',dataset='dclm',arm=arm,budget='0.8B',
                 steps=3051,removed=list(range(16,22)),stream_seed=2030,phase='F11b',loss='next_token_cross_entropy') for arm in ('P300','P1T')]

def owner(cfg):
    guard=read('/mnt/local/_gpu_guard/DISABLED');pid=os.getpid();chain=[]
    while pid>1:
        chain.append(pid);pid=process(pid)['ppid']
    if socket.gethostname()!=cfg['host'] or guard['owner_pid'] not in chain or Path(guard['output']).resolve()!=Path(cfg['output_root'],'supervised').resolve():
        raise ValueError('F11 native ownership mismatch; no signals sent')

class Queue:
    def __init__(self,cfg):
        self.cfg=cfg;self.root=Path(cfg['results_root']);self.source=Path(cfg['source_manifest']['path']).parent/'source'
        self.root.mkdir(parents=True,exist_ok=False);self.flags=[];self.events=[];self.start=time.monotonic()
        self.env={**os.environ,'PYTHONPATH':str(self.source),'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
                  'TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline',
                  'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    def flag(self,name,e):
        self.flags.append(dict(test=name,detail=str(e)));write(self.root/'flags.json',dict(flags=self.flags))
        print('F11_FLAG',name,str(e)[:300],flush=True)
    def progress(self,phase):
        write(self.root/'progress.json',dict(phase=phase,elapsed_seconds=time.monotonic()-self.start,events=self.events,flags=self.flags))
        print('F11_PHASE',phase,flush=True)
    def command(self,args,log,gpus=''):
        owner(self.cfg);p=Path(log);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as f:subprocess.run(args,cwd=self.source,env={**self.env,'CUDA_VISIBLE_DEVICES':gpus},stdout=f,stderr=subprocess.STDOUT,check=True)
    def score(self,e,gpu):
        cfg=self.cfg;common=cfg['f10']['benchmarks']['gpt_neox'];ep=self.root/'entries'/(e['model_id']+'.json');write(ep,e)
        for mode in ('nine','new'):
            out=self.root/mode/e['model_id'];prepared=common['prepared'] if mode=='nine' else cfg['f10']['prepared']
            self.command([sys.executable,'-u','-m','scripts.a2_family_f11_score','--entry',str(ep),'--output',str(out),
                '--mode',mode,'--prepared',prepared['path'],'--tokenizer',common['tokenizer'],
                '--batch-size',str(cfg['f10']['batches'][e['batch_family']]),'--generation-batch-size',str(cfg['generation_batch'])],out.parent/(e['model_id']+'.log'),str(gpu))
            if mode=='nine':
                self.command([common['eval_python'],'-u','-m','eval.a2_requests','replay','--tokenizer',common['tokenizer'],
                    '--input-spec',common['input_spec']['path'],'--dataset-root',common['dataset_root'],'--manifest',common['manifest']['path'],
                    '--tasks',*common['tasks'],'--prepared',prepared['path'],'--scored',str(out/'scores.json'),'--output',str(out/'replay.json')],out/'replay.log')
                from eval.a2_family_f9_v110 import export_items
                value,_=export_items(read(out/'replay.json'),locked(prepared),read(out/'scores.json'))
                write(out/'per_item.json',dict(value,model=e))
    def execute(self,config):
        owner(self.cfg);entries=[]
        for item in descriptors():
            name=item['run_id'];self.progress(name);require_free(list(range(8)))
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=str(sock.getsockname()[1])
            started=time.monotonic()
            try:
                self.command([sys.executable,'-u','-m','accelerate.commands.launch','--config_file','resources/accelerate_b200.yaml',
                    '--main_process_port',port,'-m','efficient_llm.a2_family_f11_heal','--config',str(config),'--run-id',name],self.root/'logs'/(name+'.log'),'0,1,2,3,4,5,6,7')
                result=read(self.root/'heals'/name/'result.json')
                self.events.append(dict(run_id=name,seconds=time.monotonic()-started,status='complete',flags=result['flags']))
                parent=next(e for e in self.cfg['entries'] if e['role']=='parent' and e['dataset']=='dclm' and e['arm']==item['arm'])
                path=Path(result['checkpoint']);hashes={str(p):sha256(p) for p in path.iterdir() if p.is_file()}
                e=dict(parent,model_id=name,role='student',parent_id=parent['model_id'],budget=.8,path=str(path),
                    file_hashes={**hashes,**{str(Path(parent['code_dir'])/k):v for k,v in parent['expected_code'].items()}},
                    final_nll=result['scores']['final']['loss'],oneshot=False,loss='next_token_cross_entropy')
                entries.append(e)
            except Exception as error:
                self.events.append(dict(run_id=name,seconds=time.monotonic()-started,status='failed',error=repr(error)));self.flag(name,repr(error))
            write(self.root/'ledger.json',dict(events=self.events,node_hours=sum(e['seconds'] for e in self.events)/3600))
        self.progress('F11b_downstream');write(self.root/'lm_entries.json',dict(entries=entries))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[(e,pool.submit(self.score,e,i)) for i,e in enumerate(entries)]
            for e,future in futures:
                try:future.result()
                except Exception as error:self.flag('downstream:'+e['model_id'],repr(error))
        self.progress('F11a_temperature')
        def lane(gpu):
            failures=[]
            for e in self.cfg['entries'][gpu::8]:
                ep=self.root/'entries'/(e['model_id']+'.json');write(ep,e)
                try:self.command([sys.executable,'-u','-m','eval.a2_family_f11_calibrate','--entry',str(ep),
                    '--output',str(self.root/'calibration'/e['model_id'])],self.root/'calibration'/(e['model_id']+'.log'),str(gpu))
                except Exception as error:failures.append((e['model_id'],repr(error)))
            return failures
        with ThreadPoolExecutor(max_workers=8) as pool:
            for failures in pool.map(lane,range(8)):
                for name,error in failures:self.flag('calibration:'+name,error)
        self.progress('report');ok=False
        try:
            from eval.a2_family_f11_report import report
            report(self.cfg,self.root);ok=read(self.root/'report_complete.json')['complete']
        except Exception as error:self.flag('report',repr(error))
        self.progress('backup')
        from scripts.a2_family_f11_backup import backup
        saved=backup(self.root,self.cfg)
        write(self.root/'verification.json',dict(report_complete=ok,backup=saved,flags=self.flags,
            compact_files={p.name:lock(p) for p in (self.root/'REPORT_FULL_v1.13.md',self.root/'F11_MODELS.csv') if p.exists()}))
        write(self.root/'complete.json',dict(execution_complete=True,report_complete=ok,backup_verified=True,flags=self.flags,elapsed_seconds=time.monotonic()-self.start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
    cfg=read(a.config)
    if cfg['spec_sha256']!=SPEC_SHA:raise ValueError('Wrong F11 spec')
    with (Path(cfg['output_root'])/'coordinator.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);Queue(cfg).execute(a.config)
        write(a.receipt,read(Path(cfg['results_root'])/'complete.json'))
