"""Admit immutable DOT-pool source/data then use unchanged native GPU supervisor."""
import argparse,json,os,shlex,shutil,socket,subprocess,sys,time
from pathlib import Path
from importlib.metadata import version
from efficient_llm.dot_pool import SPEC_SHA
from efficient_llm.dot_pool.data import sha,write
from scripts.gpu_process import inspect
from scripts.train_then_burn import BURN_HASH,GUARD_HASH
from run_experiments import load_jobs

HOST='thiennh-p6-wc69-worker-0'
BASE=Path('/mnt/local/_outputs/deep-llms_th2/dot_pool_20261112')
BURN=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f12-20261111-a01/source/resources/llm_pretrain_burn.py')
def job_config(run,prior_root=None):
 return dict(jobs=[dict(name='dot-pool-pilot',gpus=list(range(8)),argv=['{python}','-u','-m','scripts.dot_pool_queue','--root',str(run)]+(['--prior-root',str(prior_root)] if prior_root else [])+['--receipt','{run_dir}/dot-result.json'],required_outputs=[dict(path='dot-result.json',json_equals=dict(reported=True))])])

def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--name',default='a01');p.add_argument('--predecessor',choices=['a01']);a=p.parse_args()
 burn=BURN;prior_root=None
 if a.predecessor:
  old=BASE/('execution-'+a.predecessor);terminal=json.loads((old/'supervised/supervisor.json').read_text());verified=json.loads((old/'supervised/burn-verified.json').read_text())
  if not terminal.get('finished_at') or terminal.get('handoff_error') or not verified.get('collective_progress_verified'):raise ValueError('Predecessor native handoff not complete')
  burn=old/'source/resources/llm_pretrain_burn.py';prior_root=BASE/('run-'+a.predecessor)
 source=Path(__file__).resolve().parents[1];admission=BASE/('execution-'+a.name);run=BASE/('run-'+a.name);session='dot_pool_20261112_'+a.name
 if socket.gethostname()!=HOST or len(a.commit)!=40 or any(c not in '0123456789abcdef' for c in a.commit):raise ValueError('Identity mismatch')
 if admission.exists() or run.exists():raise ValueError('Existing execution, refuse duplicate')
 if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Existing session')
 if sha(source/'docs/DOT_pool_pilot_spec_v1.2.md')!=SPEC_SHA:raise ValueError('Spec mismatch')
 for package,expected in [('torch','2.14.1'),('transformers','4.57.1'),('flash-attn-4','4.0.0b33')]:
  if version(package).split('+')[0]!=expected:raise ValueError('Runtime mismatch '+package)
 for line in (source/'envs/dot_pool.txt').read_text().splitlines():
  if '==' in line:
   package,expected=line.split('==');package=package.split('[')[0]
   if version(package).split('+')[0]!=expected:raise ValueError('Dependency mismatch '+package)
 from flash_attn.cute.interface import flash_attn_varlen_func
 import torch
 if not hasattr(torch.nn.functional,'grouped_mm'):raise ValueError('Grouped GEMM missing')
 if sha(BASE/'data/ready.json')!='5c1e0213e938ad1f6a74d7c17ca67a759e2a951d9bdd4cee7cb0ee5469911340':raise ValueError('Data ready differs from retrieved prelaunch receipt')
 ready=json.loads((BASE/'data/ready.json').read_text())
 if not ready['ready'] or ready['spec_sha256']!=SPEC_SHA:raise ValueError('Data not ready')
 for sub,key in [('e1','kg_manifest_sha256'),('e2','data_manifest_sha256')]:
  if sha(BASE/'data'/sub/'manifest.json')!=ready[key]:raise ValueError('Manifest changed')
 kg=json.loads((BASE/'data/e1/manifest.json').read_text())
 for name,record in kg['files'].items():
  if sha(BASE/'data/e1'/name)!=record['sha256']:raise ValueError('KG content changed '+name)
 dm=json.loads((BASE/'data/e2/manifest.json').read_text())
 for name,record in dm['streams'].items():
  for suffix,key in [('bin','token_sha256'),('starts','starts_sha256')]:
   if sha(BASE/'data/e2'/(name+'.'+suffix))!=record[key]:raise ValueError('Packed content changed '+name+suffix)
 if sha(Path(ready['tokenizer'])/'tokenizer.json')!=dm['tokenizer_sha256']:raise ValueError('Tokenizer changed')
 if sha(burn)!=BURN_HASH or sha('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH or Path('/mnt/local/_gpu_guard/DISABLED').exists():raise ValueError('Unexpected burn/guard ownership')
 admission.mkdir();frozen=admission/'source';files=list(source.glob('*.py'))
 for d in ['efficient_llm','scripts','eval','tests']:files+=list((source/d).rglob('*.py'))
 files += [source/p for p in ['resources/llm_pretrain_burn.py','resources/downstream_english_20261023.json','resources/a2_inputs_20261028.json','docs/DOT_pool_pilot_spec_v1.2.md','envs/dot_pool.txt']]
 hashes={}
 for path in sorted(set(files)):
  rel=path.relative_to(source);target=frozen/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);hashes[str(rel)]=sha(target)
 write(admission/'source-manifest.json',dict(commit=a.commit,files=hashes))
 write(admission/'data-admission.json',dict(ready=ready,ready_sha256=sha(BASE/'data/ready.json'),physical_content_verified=True,utc=time.time(),runtime={k:version(k) for k in ['torch','transformers','numpy','flash-attn-4','nvidia-cutlass-dsl']}))
 write(admission/'inspection.json',inspect())
 # Strict runner JSON deliberately omits study metadata, which lives beside it.
 jobs=job_config(run,prior_root)
 (admission/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n');load_jobs(admission/'jobs.json')
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline','OMP_NUM_THREADS':'2','NCCL_NVLS_ENABLE':'0','PYTHONPATH':str(frozen)}
 argv=[sys.executable,'-u','-m','scripts.train_then_burn','--config',str(admission/'jobs.json'),'--output',str(admission/'supervised'),'--inspection',str(admission/'inspection.json'),'--host',HOST,'--burn-session',session+'_burn','--predecessor-burn-script',str(burn)]
 command='exec '+shlex.join(argv)+' >'+shlex.quote(str(admission/'supervisor.log'))+' 2>&1'
 subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),command],env=env,check=True)
 subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
 write(admission/'submitted.json',dict(session=session,run=str(run),commit=a.commit,utc=time.time()))
 print('DOT_QUEUE_SUBMITTED',json.dumps(dict(session=session,admission=str(admission),run=str(run))),flush=True)
if __name__=='__main__':main()
