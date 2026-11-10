"""Small read-only F12 queue/workload health; no raw scientific arrays."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json
import subprocess
from eval.a2_family_f12 import read
from efficient_llm.diagnostic_data import sha256
from scripts.gpu_process import inspect,process

def health(plan):
    cfg=read(plan);queue=Path(cfg['queue_root']);root=Path(cfg['output_root']);results=root/'results/a2_family_20261107/f12'
    print('UTC',datetime.now(timezone.utc).isoformat(),flush=True)
    for base,names in [(queue,['installed.json','ready.json','status.json']),(results,['progress.json','flags.json','ledger.json','report_complete.json','complete.json']),(root/'supervised',['supervisor.json','burn-verified.json'])]:
        for name in names:
            p=base/name
            if p.exists():print('RECORD',str(p),p.read_text()[-6000:],flush=True)
    for base in (queue,root):
        p=base/'source-manifest.json'
        if p.exists():
            m=read(p);bad=[name for name,h in m['files'].items() if sha256(base/'source'/name)!=h]
            print('F12_SOURCE_LOCKS',str(base),len(m['files']),bad,flush=True)
    for s in (cfg['queue_session'],cfg['session']):
        v=subprocess.run(['tmux','list-panes','-t',s,'-F','#{pane_pid} #{pane_dead} #{pane_dead_status}'],capture_output=True,text=True)
        print('SESSION',s,v.returncode,v.stdout.strip(),flush=True)
    for p in [queue/'queue.log',root/'supervisor.log',*sorted(results.glob('logs/*.log')),*sorted((root/'supervised/run').glob('*.log'))]:
        if p.exists():
            with p.open('rb') as f:f.seek(max(0,p.stat().st_size-1500));tail=f.read().decode(errors='replace')
            print('TAIL',str(p),tail,flush=True)
    print('COUNTS',dict(masks=len(list(results.glob('masks/P*.json'))),heals=len(list(results.glob('heals/*/result.json'))),nine=len(list(results.glob('nine/*/per_item.json'))),new=len(list(results.glob('new/*/generation_items.json')))),flush=True)
    status=inspect();print('GPU',json.dumps(status),flush=True);g=Path('/mnt/local/_gpu_guard/DISABLED');guard=read(g) if g.exists() else None
    print('GUARD',json.dumps(guard),flush=True)
    for worker in status['workers']:
        chain=[];pid=worker
        for _ in range(12):
            if pid<=1:break
            try:p=process(pid)
            except (FileNotFoundError,ProcessLookupError):break
            chain.append(p);pid=p['ppid']
        print('OWNERSHIP',worker,json.dumps(chain),'MATCH_GUARD',bool(guard and guard['owner_pid'] in [v['pid'] for v in chain]),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();health(a.plan)
