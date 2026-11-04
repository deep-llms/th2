"""Registered Why stages; scientific failures flag outputs and continue later fits."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from efficient_llm.a2_why.common import read,write,Checks,SPEC_SHA
from efficient_llm.a2_why.protocol import base_runs,descriptor,DATASETS,ARMS
from scripts.gpu_status import require_free
from scripts.a2_quant_io import lock
from scripts.a2_why_report import part1,part2,part3,final


def run_one(cfg,item,events):
    root=Path(cfg['results_root']);name=item['run_id']
    # A physical safety failure is not a scientific diagnostic: never allocate over unknown workers.
    require_free(list(range(8)))
    write(root/'progress.json',dict(active=name,completed=events))
    config=write(root/'config/runs'/(name+'.json'),dict(cfg,runs={name:item}))
    log=root/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    argv=[sys.executable,'-u','-m','accelerate.commands.launch','--config_file','resources/accelerate_b200.yaml',
        '--main_process_port',str(port),'-m','efficient_llm.a2_why.heal','--config',config['path'],'--run-id',name]
    started=time.monotonic()
    try:
        with log.open('x') as f:
            proc=subprocess.run(argv,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'},stdout=f,stderr=subprocess.STDOUT)
        result=root/'heals'/name/'result.json'
        event=dict(run_id=name,returncode=proc.returncode,result=lock(result) if result.exists() else None)
    except Exception as e:event=dict(run_id=name,returncode=-1,error=repr(e))
    event.update(seconds=time.monotonic()-started,spec_sha256=SPEC_SHA)
    events.append(event)
    with (root/'runs.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    write(root/'ledger.json',dict(events=events,elapsed_run_seconds=sum(x.get('seconds',0) for x in events),policy='report-only; no cap'))
    print('WHY_RUN_EVENT',json.dumps(event),flush=True)
    return event


def execute(cfg,stage,receipt):
    root=Path(cfg['results_root']);checks=Checks(root/'coordinator_checks.json');events=[]
    runs=base_runs()
    if stage=='yt1':
        run_one(cfg,runs[0],events)
        write(receipt,dict(execution_complete=True,events=events,coverage_complete=events[0]['returncode']==0,next='Y1 after exact data preparation; native burn handoff'))
        return
    if cfg.get('reused_yt1'):
        from scripts.a2_remaining import read_locked
        r=read_locked(cfg['reused_yt1']['result'])
        checks.record('YT1_reused_once',r['registered_run']==runs[0],cfg['reused_yt1'])
        import shutil
        src=Path(cfg['reused_yt1']['root']);dst=root/'heals'/runs[0]['run_id'];dst.mkdir(parents=True)
        # Only result/audit files; final model stays in the original YT1 root.
        for p in src.glob('*.json'):shutil.copyfile(p,dst/p.name)
        perseq=src.parents[1]/'perseq'/runs[0]['run_id']
        if perseq.exists():shutil.copytree(perseq,root/'perseq'/runs[0]['run_id'])
        event=dict(run_id=runs[0]['run_id'],returncode=0,result=cfg['reused_yt1']['result'],seconds=r.get('elapsed_seconds',0),reused=True,spec_sha256=SPEC_SHA)
        events.append(event)
        with (root/'runs.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    else:run_one(cfg,runs[0],events)
    for item in runs:
        if item['phase']=='Y1':run_one(cfg,item,events)
    n=part1(cfg)
    for item in runs:
        if item['phase']=='Y2':run_one(cfg,item,events)
    b=part2(cfg,n,post_note=False)
    for variant,r in b.items():
        if r['extension']:
            for ds in DATASETS:
                for arm in ARMS:run_one(cfg,descriptor('Y2',ds,arm,variant=variant,budget='0.8B'),events)
    part2(cfg,n,post_note=True)
    for item in runs:
        if item['phase']=='Y3':run_one(cfg,item,events)
    a=part3(cfg,n)
    for ds,r in a['extensions'].items():
        if r['trigger']:
            for arm in ARMS:run_one(cfg,descriptor('Y3',ds,arm,layers=12,budget='3.2B'),events)
        elif r['trigger'] is None:checks.record('Y3_extension_unresolved:'+ds,False,r)
    decision=checks.attempt('Why_final_report',lambda:final(cfg,events))
    write(root/'progress.json',dict(phase='Why_terminal',completed=events,next='download/replay/report then resume original Depth P300/P1T/downstream'))
    write(receipt,dict(execution_complete=True,coverage_complete=bool(decision and decision['coverage_complete']),readout_complete=decision is not None,events=events))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--receipt',required=True);p.add_argument('--stage',choices=('yt1','rest'),required=True)
    a=p.parse_args();execute(read(a.config),a.stage,a.receipt)
