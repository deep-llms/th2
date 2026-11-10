"""Closed-list F9 queue. Numerical scoring/replay use immutable v1.9 source."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from eval.a2_family_f9 import read,items,SPEC_SHA as OLD_SHA,write as old_write
from eval.a2_family_f9_v110 import SPEC_SHA,EVALUATOR_COMMIT,write,select,export_items,sampler,continuous_report
from eval.a2_request_bridge import digest
from scripts.a2_family_f9_run import Queue as Base,locked
from scripts.a2_quant_io import lock
from efficient_llm.diagnostic_data import sha256


def reuse_check(entry,scored,prepared,batch,commit):
    from eval.a2_requests import validate_scored
    validate_scored(prepared,scored)
    checks=dict(evaluator_commit=commit==EVALUATOR_COMMIT,
        identity=scored['checkpoint_identity']==entry and scored['checkpoint_identity_sha256']==digest(entry),
        batch=scored['batch_size']==batch,prepared=scored['prepared_sha256']==digest(prepared))
    if not all(checks.values()):raise ValueError('FT13 mismatch: '+str(checks))
    return checks


class Queue(Base):
    def __init__(self,cfg):
        if cfg['spec_sha256']!=SPEC_SHA or cfg['evaluator_commit']!=EVALUATOR_COMMIT:raise ValueError('Wrong v1.10 admission')
        self.admission=cfg
        compat=dict(cfg,spec_sha256=OLD_SHA,source_manifest=cfg['evaluator_source_manifest'])
        super().__init__(compat)
        self.groups,excluded=select(self.entries)
        if excluded or [len(g) for g in self.groups]!=[13,12,9,12,9]:raise ValueError('Closed list differs from reviewed 55')
        self.old=Path(cfg['prior_root']);self.evaluator=Path(cfg['evaluator_source'])
        self.batches=cfg['batches'];self.reuse=[]
        self.verify_orchestration()

    def verify_orchestration(self):
        m=locked(self.admission['source_manifest']);root=Path(self.admission['source_manifest']['path']).parent/'source'
        for name,h in m['files'].items():
            if sha256(root/name)!=h:raise ValueError('Changed orchestration source: '+name)

    def command(self,args,output,gpu=None):
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as stream:
            stream.write('F9_SCOPE_SPEC_SHA256 '+SPEC_SHA+'\nNUMERICAL_EVALUATOR '+EVALUATOR_COMMIT+'\n');stream.flush()
            code=subprocess.run(args,cwd=self.evaluator,env={**self.environment,'PYTHONPATH':str(self.evaluator),
                'A2_CODE_COMMIT':EVALUATOR_COMMIT,'CUDA_VISIBLE_DEVICES':'' if gpu is None else str(gpu)},
                stdout=stream,stderr=subprocess.STDOUT).returncode
        if code:raise RuntimeError(f'Process failed rc={code}; preserved {output}')

    def record(self,test,detail):
        with self.mutex:
            self.flags.append(dict(test=test,detail=detail));write(self.root/'flags.json',dict(flags=self.flags))
        print('F9_V110_FLAG',test,str(detail)[:300],flush=True)

    def progress(self,phase):
        from datetime import datetime,timezone
        with self.mutex:write(self.root/'progress.json',dict(phase=phase,scored=len(self.outputs),expected=len(self.entries),
            models=list(self.outputs),flags=self.flags,elapsed_seconds=time.monotonic()-self.start,utc=datetime.now(timezone.utc).isoformat()))

    def evaluate(self,entry,gpu):
        model=entry['model_id']
        if model in self.admission['unusable_ids']:
            self.record('FT12_unavailable:'+model,'Fresh physical hashes failed; excluded from scoring');return
        batch=self.batches[entry['batch_family']];common=self.cfg['benchmarks'][self.token_family(entry)]
        prepared=self.prepared[self.token_family(entry)];value=None;attempts=[]
        for folder in ('scores','FT11_batch'+str(batch)):
            source=self.old/folder/model;marker=source/'complete.json'
            if not marker.is_file():continue
            try:
                done=read(marker)
                if not done['execution_complete']:raise ValueError('Incomplete prior scoring')
                scored=locked(done['scores']);checks=reuse_check(entry,scored,prepared,batch,self.admission['evaluator_commit'])
                target=self.root/'reuse'/model/'replay.json';target.parent.mkdir(parents=True,exist_ok=True)
                if (source/'replay.json').is_file():
                    replay=read(source/'replay.json');export_items(replay,prepared,scored)
                    shutil.copyfile(source/'replay.json',target)
                else:self.replay(common,done['scores']['path'],target)
                export_items(read(target),prepared,scored)
                value=dict(scores=done['scores'],replay=lock(target),batch=batch,reused=True)
                attempts.append(dict(origin=lock(marker),checks=checks,reused=True));break
            except Exception as error:
                attempts.append(dict(origin=str(source),reused=False,error=repr(error)))
                self.record('FT13_candidate:'+model,attempts[-1])
        if value is None:value=self.score(entry,gpu,batch,'scores')
        # Additive canonical export; raw old-spec score and replay remain unchanged.
        exported,_=export_items(locked(value['replay']),prepared,locked(value['scores']))
        exported.update(model=entry,raw_scores=value['scores'],raw_replay=value['replay'],
            evaluator_commit=EVALUATOR_COMMIT,orchestration_commit=self.admission['code_commit'])
        value['items']=write(self.root/'per_item'/(model+'.json'),exported)
        with self.mutex:
            self.outputs[model]=value;self.reuse.append(dict(model=model,attempts=attempts,reused=value.get('reused',False)))
            write(self.root/'FT13.json',dict(models=self.reuse,historical_depth=locked(self.admission['old_ft13'])))
            write(self.root/'scored_index.json',dict(models=self.outputs))
        self.progress('closed_list_scoring')

    def report(self):
        from eval.a2_family_f9_report import analyze,render
        from eval.a2_family_f9_v110 import render_secondary
        replays={k:locked(v['replay']) for k,v in self.outputs.items()}
        accuracy={k:items(v) for k,v in replays.items()}
        result=analyze(self.entries,replays,draw_sampler=sampler(accuracy))
        result.update(flags=self.flags,coverage=dict(expected=len(self.entries),scored=len(replays)),
            inventory=locked(self.admission['inventory']),evaluator_commit=EVALUATOR_COMMIT,
            code_commit=self.admission['code_commit'],model_metadata=self.entries)
        write(self.root/'contrasts.json',result)
        records={k:locked(v['items'])['continuous_records'] for k,v in self.outputs.items()}
        secondary=continuous_report(self.entries,replays,records);write(self.root/'continuous.json',secondary)
        section=render(result).replace('Family v1.9 — F9 downstream','Family v1.10 — F9 downstream')
        section+='\n'+render_secondary(result,secondary)+'\nSpec SHA256: '+SPEC_SHA+'\n'
        (self.root/'contrasts.md').write_text(section)
        previous=Path(self.admission['previous_report']['path']);locked(self.admission['previous_report'])
        full=previous.read_text()+'\n'+section
        (self.root/'REPORT_FULL.md').write_text(full);(self.root.parent/'REPORT.md').write_text(full)

    def run(self):
        for e in self.entries:old_write(self.root/'entries'/(e['model_id']+'.json'),e)
        self.flags.extend(self.admission['retained_flags'])
        write(self.root/'flags.json',dict(flags=self.flags))
        for i,group in enumerate(self.groups):
            if not self.admission['spot_groups'][i]:
                candidate=next((e for e in group if e['model_id'] not in self.admission['unusable_ids']),None)
                if candidate:
                    try:self.score(candidate,0,1,'FT12_v110_spot',load_only=True,spot=True)
                    except Exception as error:self.record('FT12_group_spot:'+str(i),repr(error))
            self.progress('group_'+str(i)+'_of_5');self.lanes(group,self.evaluate)
        write(self.root/'scored_index.json',dict(models=self.outputs))
        report_ok=False
        try:self.progress('joint_bootstrap_report');self.report();report_ok=True
        except Exception as error:self.record('final_readout',repr(error))
        self.verify_source();self.verify_orchestration()
        self.progress('finished')
        write(self.root/'complete.json',dict(execution_complete=True,coverage_complete=len(self.outputs)==len(self.entries),
            report_complete=report_ok,models=len(self.outputs),expected=len(self.entries),flags=self.flags,
            elapsed_seconds=time.monotonic()-self.start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
    cfg=read(a.config);Queue(cfg).run();write(a.receipt,read(Path(cfg['downstream_root'])/'complete.json'))
