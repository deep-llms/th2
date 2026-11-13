"""Read-only, compact DOT-RL observation; no exports, signals or model loads."""
import json,time
from pathlib import Path
from efficient_llm.dot_pool.rl import sha
from scripts.gpu_process import inspect,process
from scripts.launch_dot_rl import BASE

def read(path):return json.loads(path.read_text()) if path.exists() else None
def main():
    execution=BASE/'execution-a01';run=BASE/'results/dot_rl_20261114/run-a01'
    result=dict(utc=time.time(),ownership=inspect(),submitted=read(execution/'submitted.json'),admission=read(execution/'admission.json'),supervisor=read(execution/'supervised/supervisor.json'),native=read(execution/'supervised/burn-verified.json'),status=read(run/'status.json'),budget=read(run/'budget.json'),tests=read(run/'tests_summary.json'),source_errors=[],data_errors=[],runs={},exports=[])
    sm=read(execution/'source-manifest.json')
    if sm:
        for rel,digest in sm['files'].items():
            if sha(execution/'source'/rel)!=digest:result['source_errors'].append(rel)
    data=run.parent/'data';dm=read(data/'manifest.json')
    if dm:
        for rel,r in dm['files'].items():
            if sha(data/rel)!=r['sha256']:result['data_errors'].append(rel)
    for cfg in sorted((run/'configs').glob('*.json')):
        c=read(cfg);p=Path(c['output']);result['runs'][p.name]={k:read(p/(k+'.json')) for k in ['progress','first200','train_complete','complete']}
        metrics=p/'metrics.jsonl'
        if metrics.exists():
            with metrics.open('rb') as f:
                f.seek(max(0,metrics.stat().st_size-8192));lines=f.read().decode(errors='replace').splitlines()
            result['runs'][p.name]['metrics_tail']=lines[-2:]
    result['exports']=[read(f) for f in sorted((run/'tier1').glob('*.json'))]
    result['report']=read(run/'report_complete.json')
    log=execution/'supervisor.log'
    if log.exists():
        with log.open('rb') as f:f.seek(max(0,log.stat().st_size-10000));result['supervisor_tail']=f.read().decode(errors='replace')
    print('DOT_RL_HEALTH',json.dumps(result),flush=True)
if __name__=='__main__':main()
