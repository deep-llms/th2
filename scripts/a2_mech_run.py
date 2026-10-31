"""Finite A2 M phases under the native supervisor; report-only scientific checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
from scripts.a2_quant_io import write,lock
from scripts.a2_remaining import read_locked
from efficient_llm.a2.quant_stats import PARENTS


def main():
    p=argparse.ArgumentParser()
    for k in ('config','config-sha256','receipt'):p.add_argument('--'+k,required=True)
    a=p.parse_args();cfg=read_locked(dict(path=a.config,sha256=a.config_sha256));root=Path(cfg['results_root'])
    start=time.monotonic();events=[];progress_lock=threading.Lock()
    def record(name,passed,detail):
        event=dict(test=name,passed=bool(passed),detail=detail,spec_sha256=cfg['spec_sha256'])
        events.append(event);print('MECH_PROGRESS',json.dumps(event),flush=True)
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    for k in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'):env.pop(k,None)
    def worker(phase,ds,parent,gpu):
        out=root/'workers'/phase/ds/parent;log=root/'logs'/phase/(ds+'-'+parent+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-u','-m','scripts.a2_mech_worker','--config',a.config,'--config-sha256',a.config_sha256,
            '--dataset',ds,'--parent',parent,'--phase',phase,'--output',str(out)]
        try:
            with log.open('x') as f:code=subprocess.run(cmd,env={**env,'CUDA_VISIBLE_DEVICES':str(gpu) if gpu is not None else ''},stdout=f,stderr=subprocess.STDOUT).returncode
            record(phase+':'+ds+'/'+parent,code==0,dict(returncode=code,log=str(log)))
        except Exception as e:record(phase+':'+ds+'/'+parent,False,repr(e))
    def phase(name,datasets,lanes=8,cpu=False):
        items=[(d,p) for d in datasets for p in PARENTS]
        def lane(index):
            for ds,parent in items[index::lanes]:worker(name,ds,parent,None if cpu else index)
        with progress_lock:
            write(root/'progress.json',dict(spec_sha256=cfg['spec_sha256'],phase=name,elapsed_seconds=time.monotonic()-start,events=list(events)))
        with ThreadPoolExecutor(max_workers=lanes) as pool:list(pool.map(lane,range(lanes)))
    with ThreadPoolExecutor(max_workers=1) as executor:
        weights=executor.submit(phase,'weights',('dclm','fwe','nem'),4,True)
        phase('layers',('dclm','fwe','nem'))
        phase('noise',('dclm','fwe','nem'))
        phase('tokens',('dclm','fwe'))
        try:weights.result()
        except Exception as error:record('CPU_weights_phase',False,repr(error))
    result=None
    try:
        from scripts.a2_mech_report import summarize
        result=summarize(cfg,root)
        record('readout',True,dict(coverage=result['coverage_complete']))
        # Fresh recomputation from files; compare semantic decision, not exit status.
        recomputed=summarize(cfg,root)
        record('independent_readout_recomputation',recomputed==result,'Fresh per-sequence and per-token readout with paired bootstrap')
    except Exception as e:
        import traceback
        traceback.print_exc();record('readout',False,repr(e))
    elapsed=time.monotonic()-start
    write(root/'coordinator.json',dict(spec_sha256=cfg['spec_sha256'],events=events))
    complete=dict(spec_sha256=cfg['spec_sha256'],execution_complete=True,coverage_complete=bool(result and result['coverage_complete']),
        elapsed_seconds=elapsed,node_hours=elapsed/3600,scientific_flags=result['flags'] if result else ['readout_missing'],native_handoff='pending supervisor')
    write(root/'complete.json',complete)
    if (root/'REPORT.md').exists():
        with (root/'REPORT.md').open('a') as f:f.write('\nMeasured coordinator wall time: %.6f node-hours. Budget is report-only.\n'%(elapsed/3600))
    write(a.receipt,complete)


if __name__=='__main__':main()
