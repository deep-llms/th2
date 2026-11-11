"""Read-only compact evidence for the single owned DOT-pool execution."""
import argparse,json,socket,time
from pathlib import Path
from scripts.gpu_process import inspect,process
from efficient_llm.dot_pool.data import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',default='/mnt/local/_outputs/deep-llms_th2/dot_pool_20261112');p.add_argument('--name',default='a01');a=p.parse_args();base=Path(a.base);ex=base/('execution-'+a.name);run=base/('run-'+a.name)
 result=dict(host=socket.gethostname(),utc=time.time(),gpu=inspect(),files={},progress=[],source_errors=[],logs={})
 for path in [base/'data/ready.json',base/'data/e2/progress.json',base/'data/started.json',ex/'submitted.json',ex/'data-admission.json',ex/'supervised/supervisor.json',ex/'supervised/run/run.json',ex/'supervised/burn-verified.json',run/'status.json',run/'budget.json',run/'tests.json',run/'kernel_tests.json',run/'queue_complete.json',run/'REPORT_SMALL.json',run/'manifest.json']:
  if path.is_file():result['files'][str(path)]=dict(sha256=sha(path),value=json.loads(path.read_text()))
 if (ex/'source-manifest.json').exists():
  m=json.loads((ex/'source-manifest.json').read_text());result['source_files']=len(m['files'])
  for rel,expected in m['files'].items():
   f=ex/'source'/rel
   if not f.exists() or sha(f)!=expected:result['source_errors'].append(rel)
 for path in sorted(run.glob('*/metrics.jsonl')):
  with path.open('rb') as f:
   f.seek(max(0,path.stat().st_size-4000));lines=f.read().splitlines()
  if lines:result['progress'].append(dict(run=path.parent.name,last=json.loads(lines[-1]),complete=(path.parent/'train_complete.json').exists()))
 for path in sorted(run.glob('overfit-*.json')):result['files'][path.name]=json.loads(path.read_text())
 for path in [ex/'supervisor.log']+sorted(run.glob('*.log'),key=lambda p:p.stat().st_mtime)[-5:]:
  if path.exists():
   with path.open('rb') as f:f.seek(max(0,path.stat().st_size-5000));result['logs'][str(path)]=f.read().decode(errors='replace')
 chains={}
 for pid in result['gpu']['workers']:
  seen=[];current=pid
  for _ in range(8):
   if current<=1 or current in seen:break
   seen.append(current)
   try:r=process(current);chains[str(current)]=r;current=r['ppid']
   except FileNotFoundError:break
 result['ancestry']=chains
 print('DOT_HEALTH_JSON_BEGIN');print(json.dumps(result,allow_nan=False));print('DOT_HEALTH_JSON_END',flush=True)
if __name__=='__main__':main()
