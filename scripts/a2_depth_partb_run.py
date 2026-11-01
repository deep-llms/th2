"""Registered DT4, then twelve short fits. Part C follows the posted A+B report."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from efficient_llm.a2_depth.common import read, write, Checks


def registered_runs():
    return [('dclm','P300','S1',True)]+[
        (ds,arm,block,False) for ds,block in [('dclm','W9'),('fwe','W9'),('dclm','W1'),('fwe','W1')]
        for arm in ('P300','P1T','X72')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--receipt',required=True)
    a=p.parse_args();cfg=read(a.config);root=Path(cfg['results_root']);started=time.monotonic()
    checks=Checks(root/'coordinator_checks.json');events=[]
    for ds,arm,block,dt4 in registered_runs():
        name='DT4' if dt4 else f'D-{ds}-{arm}-{block}-0.2B-lr1e-3'
        write(root/'progress.json',dict(phase='DT4' if dt4 else 'part_b',active=name,completed=events))
        log=root/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
        cmd=[sys.executable,'-u','-m','accelerate.commands.launch','--config_file','resources/accelerate_b200.yaml',
            '--main_process_port','29631','-m','efficient_llm.a2_depth.heal_short','--config',a.config,
            '--dataset',ds,'--arm',arm,'--block',block]+(['--dt4'] if dt4 else [])
        try:
            with log.open('x') as f:
                proc=subprocess.run(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'},stdout=f,stderr=subprocess.STDOUT)
            event=dict(name=name,returncode=proc.returncode,log=str(log))
        except Exception as error:
            event=dict(name=name,returncode=-1,error=repr(error),log=str(log))
        events.append(event);checks.record('process:'+name,event['returncode']==0,event)
    write(root/'ledger.json',dict(part='DT4+B',elapsed_seconds=time.monotonic()-started,
          expected_node_hours=2.7,policy='report-only',events=events))
    from scripts.a2_depth_partb_report import finalize
    decision=checks.attempt('part_b_readout',lambda:finalize(cfg))
    write(root/'progress.json',dict(phase='part_b_terminal',completed=events,
          next='publish A+B report, then DT6 and Part C with full-state resume support'))
    write(a.receipt,dict(execution_complete=True,coverage_complete=bool(decision and decision['coverage_complete']),
          readout_complete=decision is not None,elapsed_seconds=time.monotonic()-started,events=events))


if __name__=='__main__':main()
