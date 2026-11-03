"""Depth C: DT6, then three full-schedule arms and immediate saved-array notes.

Called only by the native owner after the A+B report has been posted. This
coordinator never changes frozen source or kills workers. A failed arm is resumed
from a complete matching checkpoint while progress is being made; repeated
failure at the same step is flagged and the remaining arms are attempted.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import socket
from efficient_llm.a2_depth.common import read,write,Checks
from efficient_llm.a2_depth.heal_long import run_identity
from efficient_llm.a2_depth.resume import select_checkpoint
from scripts.gpu_process import inspect
from efficient_llm.a2_depth.recovery import continued_checks, completed_arm, coordinator_lock


def require_free_gpus():
    for attempt in range(3):
        live=inspect()
        if len(live['gpus'])==8 and not any(g['pids'] for g in live['gpus']):return live
        if attempt<2:time.sleep(5)
    raise RuntimeError('Physical GPU ownership unavailable; no new workers launched')


def run_arm(cfg,config,arm,checks):
    root=Path(cfg['results_root']);out=root/'heals'/f'ST1-dclm-{arm}-S1-12.8B-lr1e-3'
    identity=run_identity(cfg,arm)
    if (out/'arm-complete.json').exists():
        try:
            marker=completed_arm(out,identity)
            checks.record('reuse_complete:'+arm,True,marker)
            return [dict(arm=arm,action='reuse verified completed arm',returncode=0)]
        except Exception as error:checks.record('reuse_complete:'+arm,False,repr(error))
    events=[];failures={};attempt=0
    while True:
        require_free_gpus()
        attempt+=1;resume=out.exists()
        log=root/'logs'/f'C-{arm}-attempt{attempt:03d}.log';log.parent.mkdir(exist_ok=True)
        # Continue numbering across an explicitly authorized native restart.
        while log.exists():
            attempt+=1;log=root/'logs'/f'C-{arm}-attempt{attempt:03d}.log'
        with socket.socket() as rendezvous:
            rendezvous.bind(('127.0.0.1',0));port=str(rendezvous.getsockname()[1])
        cmd=[sys.executable,'-u','-m','accelerate.commands.launch','--config_file','resources/accelerate_b200.yaml',
             '--main_process_port',port,'-m','efficient_llm.a2_depth.heal_long','--config',str(config),'--arm',arm]
        if resume:cmd.append('--resume')
        started=time.monotonic()
        with log.open('x') as handle:
            proc=subprocess.run(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'},stdout=handle,stderr=subprocess.STDOUT)
        event=dict(arm=arm,attempt=attempt,returncode=proc.returncode,log=str(log),elapsed_seconds=time.monotonic()-started)
        events.append(event)
        history=read(root/'attempts.json').get('events',[]) if (root/'attempts.json').exists() else []
        write(root/'attempts.json',dict(events=history+[event]))
        checks.record('process:C:'+arm,proc.returncode==0,event)
        if proc.returncode==0:
            try:
                completed_arm(out,identity)
                break
            except Exception as error:checks.record('missing_complete_arm:'+arm,False,repr(error))
        checkpoint,rejected=select_checkpoint(out/'trainer',run_identity(cfg,arm),8)
        step=int(checkpoint.name[11:]) if checkpoint else None
        checks.record('resume_candidate:'+arm,checkpoint is not None,dict(checkpoint=str(checkpoint),rejected=rejected))
        # A fresh-state retry is not an additional experiment: same arm/recipe.
        # If it cannot advance, preserve the failure and execute later arms.
        failures[step]=failures.get(step,0)+1
        if failures[step]>=3:
            checks.record('resume_no_progress:'+arm,False,dict(step=step,action='retain incomplete arm and continue remaining registered arms'))
            break
        time.sleep(min(30,10*failures[step]))
    return events


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--receipt',required=True)
    a=p.parse_args();cfg=read(a.config);root=Path(cfg['results_root'])
    with coordinator_lock(root):execute(cfg,a.config,a.receipt)


def execute(cfg,config,receipt):
    from efficient_llm.a2_depth.reconstruction import arm_order
    order=arm_order(cfg)  # Reject missing/duplicate arms before any GPU action.
    root=Path(cfg['results_root']);checks=continued_checks(root/'coordinator_checks.json')
    started=time.monotonic();events=[]
    previous=read(root/'ledger_partc.json') if (root/'ledger_partc.json').exists() else {}
    from efficient_llm.a2_depth.stream import verify
    from scripts.a2_depth_partc_report import finalize
    write(root/'execution_order.json',dict(arms=list(order),authorization=cfg.get('arm_order_authorization'),
        original_order=['P300','P1T','X72'],numerical_recipe_unchanged=True))
    dt6=checks.attempt('DT6',lambda:verify(cfg['heal_streams']['dclm']['path'],cfg['stream_history'],root,
        reconstruction=cfg.get('stream_reconstruction')))
    if cfg.get('stream_reconstruction') is not None and not (dt6 and dt6.get('passed')):
        raise ValueError('Reconstructed data failed DT6; no C arm may consume substituted bytes')
    for arm in order:
        write(root/'progress.json',dict(phase='part_c',active=arm,completed=events))
        # Physical-ownership failures propagate to the native supervisor; these
        # cannot be treated as permission to launch onto an unknown workload.
        events.extend(run_arm(cfg,config,arm,checks))
        checks.attempt('note:'+arm,lambda arm=arm:finalize(cfg,note_arm=arm))
    decision=checks.attempt('part_c_readout',lambda:finalize(cfg))
    from scripts.a2_depth_downstream import run as downstream_run
    require_free_gpus()
    downstream=checks.attempt('registered_downstream',lambda:downstream_run(cfg,config))
    write(root/'ledger_partc.json',dict(elapsed_seconds=previous.get('elapsed_seconds',0)+time.monotonic()-started,events=previous.get('events',[])+events,policy='report-only',attempt_history='attempts.json'))
    write(root/'progress.json',dict(phase='part_c_terminal',completed=events,next='external saved-array verification and final native handoff'))
    write(receipt,dict(execution_complete=True,training_coverage_complete=bool(decision and decision['coverage_complete']),
                        downstream_complete=bool(downstream and downstream['coverage_complete']),
                        readout_complete=decision is not None,events=events))
    def report():
        import json
        path=root/'REPORT.md';training=root/'REPORT_PART_C.md'
        text=training.read_text() if training.exists() else '# Depth C\n\nTraining readout unavailable; see checks.\n'
        text='\n'.join(line for line in text.splitlines() if not line.startswith('Remaining:'))+'\n'
        prior=cfg.get('part_ab_report_local')
        if prior:text=Path(prior).read_text()+'\n\n'+text
        text+='\n\n## Part C downstream (reported only)\n\n```json\n'+json.dumps(downstream,indent=2)+'\n```\n'
        text+='\nIndependent external saved-array verification and native terminal handoff still require separate evidence.\n'
        path.write_text(text)
        write(root/'decision_depth.json',dict(training=decision,downstream=downstream,
             coverage_complete=bool(decision and decision['coverage_complete'] and downstream and downstream['coverage_complete']),
             flags=[e for e in checks.items if not e['passed']]))
    checks.attempt('combined_report',report)


if __name__=='__main__':main()
