"""F9 evaluation-only queue, called inside the existing native GPU supervisor.

No training, downloads, process signals or global environment changes. All GPU
children have one explicit lane. Scientific failures are retained per model.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import numpy as np
from eval.a2_family_f9 import SPEC_SHA,read,write,items,batch_decision
from eval.a2_request_bridge import digest
from efficient_llm.diagnostic_data import sha256
from scripts.a2_quant_io import lock


def locked(value):
    if sha256(value['path'])!=value['sha256']:raise ValueError('Changed locked input: '+value['path'])
    return read(value['path'])


class Queue:
    def __init__(self,cfg):
        self.cfg=cfg;self.root=Path(cfg['downstream_root']);self.start=time.monotonic()
        self.entries=locked(cfg['inventory'])['models'];self.flags=[];self.outputs={}
        for entry in self.entries:entry['spec_sha256']=SPEC_SHA
        self.mutex=threading.Lock();self.batches={};self.prepared={}
        self.verify_source()
        for family,common in cfg['benchmarks'].items():
            self.prepared[family]=locked(common['prepared'])
            for key in ('input_spec','manifest'):locked(common[key])
        self.environment={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
            'TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline',
            'TOKENIZERS_PARALLELISM':'false','CUDA_DEVICE_ORDER':'PCI_BUS_ID','OMP_NUM_THREADS':'1'}
        for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'):self.environment.pop(key,None)
        if cfg['spec_sha256']!=SPEC_SHA or sorted(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))!=list(range(8)):
            raise ValueError('Wrong spec or GPU allocation')

    def verify_source(self):
        manifest=locked(self.cfg['source_manifest']);root=Path(self.cfg['source_manifest']['path']).parent/'source'
        for name,h in manifest['files'].items():
            if sha256(root/name)!=h:raise ValueError('Frozen source changed: '+name)

    def record(self,test,detail):
        with self.mutex:
            self.flags.append(dict(test=test,detail=detail))
            write(self.root/'flags.json',dict(flags=self.flags))
        print('F9_FLAG',test,str(detail)[:300],flush=True)

    def command(self,args,output,gpu=None):
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as stream:
            stream.write('F9_SPEC_SHA256 '+SPEC_SHA+'\n');stream.flush()
            code=subprocess.run(args,env={**self.environment,'CUDA_VISIBLE_DEVICES':'' if gpu is None else str(gpu)},
                stdout=stream,stderr=subprocess.STDOUT).returncode
        if code:raise RuntimeError(f'Process failed rc={code}; preserved {output}')

    def token_family(self,entry):return 'smollm3' if entry['family']=='smollm3' else 'gpt_neox'

    def score(self,entry,gpu,batch,label,load_only=False,spot=False):
        root=self.root/label/entry['model_id'];common=self.cfg['benchmarks'][self.token_family(entry)]
        epath=self.root/'entries'/(entry['model_id']+'.json')
        args=[sys.executable,'-u','-m','eval.a2_family_f9_score','--entry',str(epath),
              '--prepared',common['prepared']['path'],'--output',str(root),'--batch-size',str(batch)]
        if load_only:args.append('--load-only')
        if spot:args.append('--spot-check')
        self.command(args,root.parent/(entry['model_id']+'.log'),gpu)
        done=read(root/'complete.json')
        if not done['execution_complete']:raise ValueError('Incomplete scorer receipt')
        if spot:
            proof=read(root/'FT12_spot.json')
            if not proof.get('passed'):self.record('FT12_CulturaX:'+entry['model_id'],proof)
        if load_only:return lock(root/'load.json')
        score=locked(done['scores'])
        if score['checkpoint_identity']!=entry or score['checkpoint_identity_sha256']!=digest(entry):
            raise ValueError('Scored model identity changed')
        target=root/'replay.json';self.replay(common,done['scores']['path'],target)
        value=read(target);items(value)
        return dict(scores=done['scores'],replay=lock(target),batch=batch)

    def replay(self,common,scored,target):
        args=[common['eval_python'],'-u','-m','eval.a2_requests','replay',
              '--tokenizer',common['tokenizer'],'--input-spec',common['input_spec']['path'],
              '--dataset-root',common['dataset_root'],'--manifest',common['manifest']['path'],
              '--tasks',*common['tasks'],'--prepared',common['prepared']['path'],
              '--scored',str(scored),'--output',str(target)]
        if common.get('tokenizer_family')=='smollm3':
            args+=['--tokenizer-family','smollm3','--tokenizer-manifest',common['tokenizer_manifest']['path']]
        self.command(args,str(target)+'.log')
        write(target,read(target))

    def lanes(self,entries,action):
        def lane(gpu):
            for entry in entries[gpu::8]:
                try:action(entry,gpu)
                except Exception as error:self.record(action.__name__+':'+entry['model_id'],repr(error))
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(lane,range(8)))

    def progress(self,phase):
        with self.mutex:
            write(self.root/'progress.json',dict(phase=phase,scored=len(self.outputs),inventory=len(self.entries),
                models=list(self.outputs),elapsed_seconds=time.monotonic()-self.start,
                utc=datetime.now(timezone.utc).isoformat(),flags=self.flags))

    def report(self,name):
        from eval.a2_family_f9_report import analyze,render
        replays={key:locked(value['replay']) for key,value in self.outputs.items()}
        result=analyze(self.entries,replays)
        result.update(flags=self.flags,coverage=dict(expected=len(self.entries),scored=len(replays)),
                      elapsed_seconds=time.monotonic()-self.start,code_commit=self.cfg['code_commit'],
                      inventory=locked(self.cfg['inventory']),model_metadata=self.entries,
                      earlier_downstream=self.cfg.get('earlier_downstream',[]))
        write(self.root/(name+'.json'),result)
        (self.root/(name+'.md')).write_text(render(result)+'\nSpec SHA256: '+SPEC_SHA+'\n')
        return result

    def run(self):
        loaded={};parts=set()
        for e in self.entries:
            write(self.root/'entries'/(e['model_id']+'.json'),e)
            # One complete CulturaX final-set rerun for each original part.
            part=e['part'];e_spot=part not in parts;parts.add(part)
            e['_spot_selected']=e_spot
        # Do not mutate the identity passed to the scorer.
        spots={e['model_id']:e.pop('_spot_selected') for e in self.entries}
        def inventory_load(e,g):
            proof=self.score(e,g,1,'FT12_loads',load_only=True,spot=spots[e['model_id']])
            with self.mutex:loaded[e['model_id']]=proof
        self.progress('FT12_load_all_before_downstream');self.lanes(self.entries,inventory_load)
        write(self.root/'FT12_loads.json',dict(loaded=loaded,expected=len(self.entries),
            all_load= len(loaded)==len(self.entries),spots=spots,flags=self.flags))
        first={}
        for e in self.entries:
            if e['model_id'] in loaded:first.setdefault(e['batch_family'],e)
        candidates=[(e,b) for e in first.values() for b in (1,2)];trials={}
        def trial(item):
            gpu,(e,b)=item
            try:trials[(e['model_id'],b)]=self.score(e,gpu,b,'FT11_batch'+str(b))
            except Exception as error:self.record('FT11_trial:'+e['model_id']+':'+str(b),repr(error))
        self.progress('FT11_full_nine_tasks')
        with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(trial,enumerate(candidates)))
        for family,e in first.items():
            try:
                one,two=(trials[(e['model_id'],b)] for b in (1,2))
                decision=batch_decision(locked(one['replay']),locked(two['replay']))
                a=np.asarray(locked(one['scores'])['scores'],dtype=float)
                b=np.asarray(locked(two['scores'])['scores'],dtype=float)
                diff=np.abs(a[:,0]-b[:,0]);np.save(self.root/('FT11_'+family+'_request_differences.npy'),diff)
                decision.update(max_request_LL_difference=float(diff.max()),
                    mean_request_LL_difference=float(diff.mean()),changed_requests=int(np.count_nonzero(diff)),
                    changed_greedy=int(np.count_nonzero(a[:,1]!=b[:,1])),model=e['model_id'])
                if np.any(diff):self.record('FT11_request_numerical_differences:'+family,decision)
                if not decision['passed']:self.record('FT11_accuracy_fallback:'+family,decision)
                self.batches[family]=decision['batch_size']
                self.outputs[e['model_id']]=trials[(e['model_id'],decision['batch_size'])]
            except Exception as error:
                decision=dict(passed=False,batch_size=1,error=repr(error));self.batches[family]=1
                self.record('FT11_unavailable_use_batch1:'+family,decision)
            write(self.root/('FT11_'+family+'.json'),decision)
        # Exact reuse is deliberately gated by the original scorer/bridge commit.
        # A changed evaluator requires fresh scores; original evidence is kept.
        reuse=[]
        for candidate in self.cfg.get('depth_reuse',[]):
            checks={};detail={}
            try:
                old=locked(candidate['scores']);oldreplay=locked(candidate['replay'])
                prepared=self.prepared['gpt_neox'];protocol=prepared['protocol']
                checks=dict(code_commit=candidate['code_commit']==self.cfg['code_commit'],
                    tasks_metrics=oldreplay['a2_metadata']['protocol']['tasks']==protocol['tasks'],
                    settings=all(oldreplay['a2_metadata']['protocol'][k]==protocol[k] for k in ('num_fewshot','seed','max_length','limit')),
                    prepared=old['prepared_sha256']==digest(prepared),
                    batch=old['batch_size']==self.batches.get('opensci',1),per_item=True)
                items(oldreplay)
                if all(checks.values()):
                    entry=next(e for e in self.entries if e['model_id']==candidate['model_id'])
                    checks['weights']=all(h in json_weight_hashes(old['checkpoint_identity']) for p,h in entry['file_hashes'].items() if p.endswith(('.safetensors','.bin')))
                if checks and all(checks.values()):self.outputs[candidate['model_id']]=dict(scores=candidate['scores'],replay=candidate['replay'],batch=old['batch_size'],reused=True)
            except Exception as error:checks['artifact_validation']=False;detail['error']=repr(error)
            reused=bool(checks) and all(checks.values())
            reuse.append(dict(model=candidate['model_id'],checks=checks,reused=reused,detail=detail,original=candidate))
            if not reused:self.record('FT13_rescore_preserve_original:'+candidate['model_id'],checks)
        write(self.root/'FT13.json',dict(models=reuse,expected_six=len(reuse)==6))
        for phase in (1,2,3):
            self.progress('F9_order_'+str(phase))
            todo=[e for e in self.entries if e['phase']==phase and e['model_id'] in loaded and e['model_id'] not in self.outputs]
            def score_model(e,g):
                value=self.score(e,g,self.batches.get(e['batch_family'],1),'scores')
                with self.mutex:self.outputs[e['model_id']]=value
                self.progress('F9_order_'+str(phase))
            self.lanes(todo,score_model)
            if phase==2:
                try:self.report('INTERIM');print('F9_INTERIM_READY',self.root/'INTERIM.md',flush=True)
                except Exception as error:self.record('interim_readout',repr(error))
        write(self.root/'scored_index.json',dict(models=self.outputs))
        report_ok=False
        try:
            self.report('contrasts')
            previous=Path(self.cfg['previous_report']['path'])
            if sha256(previous)!=self.cfg['previous_report']['sha256']:raise ValueError('Original report changed')
            text=previous.read_text()+'\n'+(self.root/'contrasts.md').read_text()
            (self.root.parent/'REPORT.md').write_text(text)
            (self.root/'REPORT_FULL.md').write_text(text)
            report_ok=True
        except Exception as error:self.record('final_readout',repr(error))
        self.progress('F9_scoring_finished')
        self.verify_source()
        write(self.root/'complete.json',dict(execution_complete=True,coverage_complete=len(self.outputs)==len(self.entries),
            report_complete=report_ok,models=len(self.outputs),expected=len(self.entries),flags=self.flags,
            elapsed_seconds=time.monotonic()-self.start))


def json_weight_hashes(value):
    if isinstance(value,dict):return {x for v in value.values() for x in json_weight_hashes(v)}
    if isinstance(value,list):return {x for v in value for x in json_weight_hashes(v)}
    return {value} if isinstance(value,str) and len(value)==64 else set()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
    cfg=read(a.config);Queue(cfg).run()
    write(a.receipt,read(Path(cfg['downstream_root'])/'complete.json'))
