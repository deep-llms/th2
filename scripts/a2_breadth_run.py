"""Sequential four eight-rank heals, then bounded eight-lane SmolLM3 evaluation."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import time
from efficient_llm.a2_breadth.common import read,write,Checks

def ready(entry):
    return all((Path(entry['path'])/f['name']).is_file() and (Path(entry['path'])/f['name']).stat().st_size==f['bytes'] for f in entry['files'])
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--receipt',required=True);a=p.parse_args();cfg=read(a.config)
    root=Path(cfg['results_root']);started=time.monotonic();checks=Checks(root/'coordinator_checks.json');events=[]
    def call(name,cmd,gpus):
        path=root/'logs'/(name+'.log');path.parent.mkdir(exist_ok=True)
        with path.open('x') as log:
            proc=subprocess.run(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpus},stdout=log,stderr=subprocess.STDOUT)
        return dict(name=name,returncode=proc.returncode,log=str(path))
    for ds,arm in [('dclm','S72'),('dclm','S188'),('fwe','S72'),('fwe','S188')]:
        name=f'ST1-{ds}-{arm}-S1-0.2B-lr1e-3';write(root/'progress.json',dict(phase='heal',active=name,completed=events))
        cmd=[sys.executable,'-u','-m','accelerate.commands.launch','--config_file','resources/accelerate_b200.yaml','--main_process_port','29631','-m','efficient_llm.a2_breadth.heal','--config',a.config,'--dataset',ds,'--arm',arm]
        event=call(name,cmd,'0,1,2,3,4,5,6,7');events.append(event);checks.record('process:'+name,event['returncode']==0,event)
    # No time cap: background controller transfers can outlast the four fits.
    entries=cfg['downloads']['checkpoints'];wait_started=time.monotonic()
    while not all(ready(e) for e in entries):
        write(root/'progress.json',dict(phase='waiting_registered_downloads',ready=[e['revision'] for e in entries if ready(e)],completed=events,wait_seconds=time.monotonic()-wait_started))
        time.sleep(30)
    download_wait=time.monotonic()-wait_started
    event=call('prepare_documents',[sys.executable,'-u','-m','efficient_llm.a2_breadth.smol','--config',a.config,'--prepare'],'');events.append(event);checks.record('prepare_documents',event['returncode']==0,event)
    write(root/'progress.json',dict(phase='smollm3_evaluation',completed=events))
    def lane(gpu):
        results=[]
        for index in range(gpu,len(entries),8):
            name=entries[index]['revision']
            try:results.append(call(name,[sys.executable,'-u','-m','efficient_llm.a2_breadth.smol','--config',a.config,'--index',str(index)],str(gpu)))
            except Exception as e:results.append(dict(name=name,returncode=-1,error=repr(e)))
        return results
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for results in pool.map(lane,range(8)):
            for event in results:events.append(event);checks.record('process:'+event['name'],event['returncode']==0,event)
    write(root/'ledger.json',dict(elapsed_seconds=time.monotonic()-started,download_wait_seconds=download_wait,estimated_node_hours=2,policy='report-only',events=events))
    from scripts.a2_breadth_report import finalize
    decision=checks.attempt('CPU_readout',lambda:finalize(cfg))
    write(root/'progress.json',dict(phase='terminal',completed=events))
    write(a.receipt,dict(execution_complete=True,coverage_complete=bool(decision and decision['coverage_complete']),readout_complete=decision is not None,elapsed_seconds=time.monotonic()-started,events=events))
if __name__=='__main__':main()
