"""Registered Family v1.3 stages; failed scientific runs do not cancel later fits."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from efficient_llm.a2_family.common import read,write,Checks,SPEC_SHA
from efficient_llm.a2_family.protocol import descriptor,grid,choose_lr,changed,mixture_reading
from scripts.gpu_status import require_free


def launch(cfg,module,name,arguments,events):
    root=Path(cfg['results_root'])
    for attempt in range(3):
        try:require_free(list(range(8)));break
        except RuntimeError:
            if attempt==2:raise
            time.sleep(5)
    if any(e['name']==name for e in events):raise ValueError('Duplicate job identity')
    write(root/'progress.json',dict(active=name,events=events))
    log=root/'logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=str(sock.getsockname()[1])
    argv=[sys.executable,'-u','-m','accelerate.commands.launch','--config_file','resources/accelerate_b200.yaml','--main_process_port',port,'-m',module,*arguments]
    started=time.monotonic()
    with log.open('x') as out:
        result=subprocess.run(argv,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'},stdout=out,stderr=subprocess.STDOUT)
    event=dict(name=name,module=module,returncode=result.returncode,seconds=time.monotonic()-started,log=str(log));events.append(event)
    with (root/'runs.jsonl').open('a') as f:f.write(json.dumps(dict(event,spec_sha256=SPEC_SHA))+'\n')
    write(root/'ledger.json',dict(events=events,node_hours=sum(e['seconds'] for e in events)/3600,policy='estimate/report only; no deadline'))
    print('FAMILY_RUN_EVENT',json.dumps(event),flush=True)
    return event


def finite(x):return isinstance(x,(int,float)) and math.isfinite(x)


def primary_contrast(decision,budget,lr,seed=2030,data='fwe',arms=('k6','k20')):
    quantity={('k6','k20'):'E',('k20','k64'):'D64'}[arms]
    basis=decision.get('overlap',{}).get(data,{}).get('primary')
    if basis not in ('full','reduced'):return None
    for pair in decision.get('pairs',[]):
        if (pair['quantity'],pair['budget'],pair['lr'],pair['seed'],pair['data'])==(quantity,budget,lr,seed,data):
            return pair[basis]['E']
    return None


def execute(cfg,config,receipt):
    root=Path(cfg['results_root']);events=[];checks=Checks(root/'coordinator_checks.json');completed=[];latest_readout={};trigger_records=[]
    for ds in ('dclm','fwe'):launch(cfg,'efficient_llm.a2_family.reference','F0-'+ds,['--config',str(config),'--dataset',ds],events)
    for arm in ('k6','k20','k64'):launch(cfg,'efficient_llm.a2_family.checks','F1-'+arm,['--config',str(config),'--arm',arm],events)
    caches={a:read(root/'parents'/a/'result.json') for a in ('k6','k20','k64') if (root/'parents'/a/'result.json').is_file()}
    cfg=dict(cfg,mask_documents=all(c.get('mask_documents') for c in caches.values()) and len(caches)==3)
    runtime=write(root/'config/healing.json',cfg)
    def fit(item):
        # The same arm/data/budget/LR/order may already exist under another stage.
        key=lambda r:tuple(r[k] for k in ('arm','data','budget','lr','stream_seed'))
        same=[r for r in completed if key(r)==key(item)]
        if same:return
        item_lock=write(root/'config/runs'/(item['run_id']+'.json'),item)
        launch(cfg,'efficient_llm.a2_family.heal',item['run_id'],['--config',runtime['path'],'--item',item_lock['path']],events)
        completed.append(item)
        checks.attempt('partial_readout',lambda:report())
    def scalar(arm,budget,lr,seed=2030,data='fwe',metric='delta_post'):
        for item in completed:
            if (item['arm'],item['budget'],item['lr'],item['stream_seed'],item['data'])==(arm,budget,lr,seed,data):
                p=root/'heals'/item['run_id']/'result.json'
                if p.is_file():
                    value=read(p)
                    return value.get('scores',{}).get('tuning',{}).get('loss') if metric=='tuning' and value.get('scores',{}).get('tuning') else value.get(metric)
        return None
    def contrast(budget,lr,seed=2030,data='fwe',arms=('k6','k20')):
        value=primary_contrast(latest_readout,budget,lr,seed,data,arms)
        trigger_records.append(dict(budget=budget,lr=lr,seed=seed,data=data,arms=arms,value=value,overlap_basis=latest_readout.get('overlap',{}).get(data),completed_runs=len(completed)))
        return value
    def report(final=False):
        nonlocal latest_readout
        from scripts.a2_family_report import report as build
        latest_readout=build(cfg,events,final=final)
        return latest_readout
    for item in grid(cfg['peak_lr']):fit(item)
    tuning={(arm,lr):scalar(arm,'0.2B',lr,metric='tuning') for arm in ('k6','k20') for lr in (cfg['peak_lr']/8,cfg['peak_lr']/4,cfg['peak_lr']/2)}
    selected=choose_lr(tuning,cfg['peak_lr'])
    if selected.get('extend_down'):
        for item in grid(cfg['peak_lr'],extension=True):fit(item)
        for arm in ('k6','k20'):tuning[arm,cfg['peak_lr']/16]=scalar(arm,'0.2B',cfg['peak_lr']/16,metric='tuning')
        selected=choose_lr(tuning,cfg['peak_lr'])
    selected['per_arm_best']={a:min(((lr,v) for (arm,lr),v in tuning.items() if arm==a and finite(v)),key=lambda x:(x[1],x[0]),default=None) for a in ('k6','k20')}
    write(root/'lr_selection.json',selected);lr=selected['lr']
    for arm in ('k6','k20'):fit(descriptor('F2',arm,'0.2B',lr,seed=2031))
    for arm in ('k6','k20'):fit(descriptor('F2c',arm,'0.2B',lr,data='mix'))
    for arm in ('k6','k20'):fit(descriptor('F3',arm,'0.8B',lr))
    e02,e08=contrast('0.2B',lr),contrast('0.8B',lr)
    f3=changed(e02,e08) if finite(e02) and finite(e08) else None
    checks.record('F3_trigger_evaluable',f3 is not None,dict(E02=e02,E08=e08))
    if f3:
        for arm in ('k6','k20'):fit(descriptor('F3',arm,'3.2B',lr))
    for budget in ('0.2B','0.8B'):fit(descriptor('F4','k64',budget,lr))
    d02,d08=[contrast(b,lr,arms=('k20','k64')) for b in ('0.2B','0.8B')]
    f4=bool(f3) or (changed(d02,d08) if finite(d02) and finite(d08) else None)
    checks.record('F4_trigger_evaluable',f4 is not None,dict(D02=d02,D08=d08,F3=f3))
    if f4:
        fit(descriptor('F4','k64','3.2B',lr))
        if not f3:fit(descriptor('F4','k20','3.2B',lr))
    e02=contrast('0.2B',lr)
    mix,second=contrast('0.2B',lr,data='mix'),contrast('0.2B',lr,seed=2031)
    fe4=mixture_reading(mix,e02,second) if all(finite(x) for x in (mix,e02,second)) else dict(FE4='unverified',run_F5=None)
    # FE4 trigger itself does not depend on availability of the secondary noise value.
    if finite(mix) and finite(e02) and not finite(second):fe4=dict(FE4='data-dependent' if abs(mix-e02)>=.003 else 'robust',run_F5=abs(mix-e02)>=.003,data_order_difference=None)
    if fe4['run_F5']:
        for arm in ('k6','k20'):fit(descriptor('F5',arm,'0.8B',lr,data='mix'))
    write(root/'extensions.json',dict(F3=f3,F4=f4,F5=fe4['run_F5'],FE4=fe4,trigger_records=trigger_records,trigger_policy='Primary overlap-adjudicated quantity using evidence available at each registered trigger; records immutable after later runs',missing_trigger_policy='report unverified and continue independent registered stages; no invented measurement'))
    decision=checks.attempt('final_readout',lambda:report(final=True))
    backup=None
    if decision is not None:
        from scripts.a2_family_backup import backup as save_backup
        backup=checks.attempt('scientific_backup',lambda:save_backup(root))
    write(receipt,dict(backup=backup,execution_complete=True,readout_complete=decision is not None,coverage_complete=all(x['returncode']==0 for x in events),events=events,next='Verify scientific backup and resume authorized Depth from pinned native state'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--receipt',required=True);a=p.parse_args();cfg=read(a.config)
    if cfg['spec_sha256']!=SPEC_SHA:raise ValueError('Wrong Family config')
    root=Path(cfg['results_root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(cfg,a.config,a.receipt)
