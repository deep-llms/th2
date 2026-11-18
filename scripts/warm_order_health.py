"""Read-only bounded pilot health; no signals, relaunch or network."""
import json,os,socket,time
from pathlib import Path
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect,process

BASE=Path('/mnt/local/_outputs/deep-llms_th2/warm_order_20261118')
def read(p):return json.loads(p.read_text()) if p.exists() else None
def tail(p,size=5000):
    if not p.exists():return ''
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-size));return f.read().decode(errors='replace')
def main():
    base=BASE;ex=base/'execution';out=dict(utc=time.time(),host=socket.gethostname(),ownership=inspect(),submitted=read(ex/'submitted.json'),admission=read(ex/'admission.json'),probe=read(base/'gpu-probe.json'),status=read(base/'status.json'),failure=read(base/'failure.json'),terminal=read(base/'terminal.json'),native=read(base/'native-verified.json'),runs={},source_errors=[])
    manifest=read(ex/'source.json') or {}
    for rel,h in manifest.items():
        if sha(ex/'source'/rel)!=h:out['source_errors'].append(rel)
    for name in ['D-s0','D-s1','R-s0','R-s1','W-s0','W-s1']:
        run=base/'runs'/name;lines=tail(run/'trace.jsonl').splitlines();last=None
        if lines:
            try:last=json.loads(lines[-1])
            except ValueError:pass
        result=dict(complete=read(run/'complete.json'),checkpoint_metadata=read(run/'checkpoint.json'),last_trace=last,eval_complete=read(base/'eval'/name/'complete.json'),eval_report=read(base/'eval'/name/'report.json'))
        if result['complete']:
            result['actual_terminal_hash_ok']={file:sha(run/file)==result['complete'][key] for file,key in [('weights.pt','weights_sha256'),('fullstate.pt','fullstate_sha256')]}
        out['runs'][name]=result
    out['logs']={p.name:tail(p,3000) for p in [ex/'queue.log',base/'probe-R-s0.log',base/'launch-tests.log']}
    if (base/'failure.json').exists():out['failure_logs']={p.name:tail(p,2000) for p in base.glob('*.log')}
    out['worker_ancestry']={}
    for pid in out['ownership']['workers']:
        chain=[];seen=set()
        while pid>1 and pid not in seen and len(chain)<6:
            seen.add(pid)
            try:r=process(pid)
            except (FileNotFoundError,ProcessLookupError):break
            chain.append(r);pid=r['ppid']
        out['worker_ancestry'][str(chain[0]['pid']) if chain else str(pid)]=chain
    guard=Path('/mnt/local/_gpu_guard/DISABLED');out['guard_hold']=read(guard) if guard.exists() else None
    if (base/'backup.json').exists():out['backup']=read(base/'backup.json')
    print('WARM_ORDER_HEALTH',json.dumps(out,allow_nan=False),flush=True)
if __name__=='__main__':main()
