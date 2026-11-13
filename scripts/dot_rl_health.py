"""Read-only, compact DOT-RL observation; no exports, signals or model loads."""
import hashlib,json,sys,time
from pathlib import Path
from efficient_llm.dot_pool.rl import sha
from scripts.gpu_process import inspect,process
from scripts.launch_dot_rl import BASE

def read(path):return json.loads(path.read_text()) if path.exists() else None
def tail_jsonl(path):
    if not path.exists():return []
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-32768));lines=f.read().decode(errors='replace').splitlines()
    records=[]
    for line in lines:
        try:records.append(json.loads(line))
        except json.JSONDecodeError:pass  # Partial first/last line while the worker appends.
    return records[-2:]

def main():
    execution=BASE/'execution-a01';run=BASE/'results/dot_rl_20261114/run-a01'
    result=dict(utc=time.time(),ownership=inspect(),submitted=read(execution/'submitted.json'),admission=read(execution/'admission.json'),supervisor=read(execution/'supervised/supervisor.json'),native=read(execution/'supervised/burn-verified.json'),status=read(run/'status.json'),budget=read(run/'budget.json'),tests=read(run/'tests_summary.json'),source_errors=[],data_errors=[],runs={},exports=[])
    result['source_manifest_sha256']=sha(execution/'source-manifest.json')
    expected={}
    def pin(argv,name):expected[hashlib.sha256(b'\0'.join(x.encode() for x in argv)+b'\0').hexdigest()]=name
    for arm,gpu in [('L',0),('P-rand',2),('J-RL',4)]:
        pin([sys.executable,'-u','-m','scripts.dot_rl_checks','--arm',arm,'--data',str(run.parent/'data'),'--output',str(run/('checks-'+arm))],'checks-'+arm)
    for cfg in sorted((run/'configs').glob('*.json')):
        pin([sys.executable,'-u','-m','efficient_llm.dot_pool.rl_run','--config',str(cfg)],cfg.stem)
    result['worker_commands']={str(pid):expected.get(record['command_sha256'],'UNMATCHED') for pid in result['ownership']['workers'] for record in [result['ownership']['processes'][str(pid)]]}
    ancestors={}
    for pid in result['ownership']['workers']:
        chain=[]
        for _ in range(7):
            if pid<=1:break
            try:r=process(pid)
            except (ProcessLookupError,FileNotFoundError):break
            chain.append(r);pid=r['ppid']
        ancestors[str(chain[0]['pid']) if chain else 'exited']=chain
    result['ancestry']=ancestors
    result['partial_checks']={p.parent.name:read(p) for p in run.glob('checks-*/checks.json')}
    sm=read(execution/'source-manifest.json')
    if sm:
        for rel,digest in sm['files'].items():
            if sha(execution/'source'/rel)!=digest:result['source_errors'].append(rel)
    data=run.parent/'data';dm=read(data/'manifest.json')
    if dm:
        result['data_manifest_sha256']=sha(data/'manifest.json')
        for rel,r in dm['files'].items():
            if sha(data/rel)!=r['sha256']:result['data_errors'].append(rel)
    for cfg in sorted((run/'configs').glob('*.json')):
        c=read(cfg);p=Path(c['output']);result['runs'][p.name]={k:read(p/(k+'.json')) for k in ['progress','first200','train_complete','complete']}
        result['runs'][p.name]['curves_tail']=tail_jsonl(p/'curves.jsonl')
        result['runs'][p.name]['checkpoints']=[dict(path=str(f.parent),manifest=read(f),manifest_sha256=sha(f),mode='metadata; state hash recorded at save, not rehashed in this health') for f in sorted(p.glob('checkpoint-*/complete.json'))]
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
