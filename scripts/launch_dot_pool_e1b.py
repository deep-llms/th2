"""Admit v1.5 E1b, preserved pilot inputs, exact native ownership; no E2 runs."""
import argparse,gzip,hashlib,json,os,shlex,shutil,socket,subprocess,sys,time
from pathlib import Path
from importlib.metadata import version
from efficient_llm.dot_pool import SPEC_SHA, E1B_SPEC_SHA, PILOT_SPEC_SHA
from efficient_llm.dot_pool.data import sha,write
from scripts.gpu_process import inspect
from scripts.train_then_burn import BURN_HASH,GUARD_HASH
from run_experiments import load_jobs

HOST='thiennh-p6-wc69-worker-0'
BASE=Path('/mnt/local/_outputs/deep-llms_th2/dot_pool_20261112')
BURN=BASE/'sr-recheck-a01/source/resources/llm_pretrain_burn.py'
def pilot_reference(source,pin):
 reference=Path(source)/'resources/dot_pool_v12_reference.md.gz'
 raw=gzip.decompress(reference.read_bytes())
 if hashlib.sha256(raw).hexdigest()!=pin['pilot_report_sha256']:raise ValueError('Pilot report changed')
 return raw
def job_config(run):
 return dict(jobs=[dict(name='dot-pool-e1b-v15',gpus=list(range(8)),argv=['{python}','-u','-m','scripts.dot_pool_e1b_queue','--root',str(run),'--receipt','{run_dir}/dot-result.json'],required_outputs=[dict(path='dot-result.json',json_equals=dict(reported=True))])])

def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args()
 burn=BURN
 if SPEC_SHA!=E1B_SPEC_SHA:raise ValueError('v1.5 process protocol not selected')
 source=Path(__file__).resolve().parents[1];admission=BASE/'execution-e1b-v15-a02';run=BASE/'run-e1b-v15-a02';session='dot_pool_e1b_v15_a02'
 if socket.gethostname()!=HOST or len(a.commit)!=40 or any(c not in '0123456789abcdef' for c in a.commit):raise ValueError('Identity mismatch')
 # a01 failed while freezing the reference, before the supervisor was submitted.
 for old in [BASE/'run-e1b-v15-a01',BASE/'execution-e1b-v15-a01/submitted.json',BASE/'execution-e1b-v15-a01/supervised']:
  if old.exists():raise ValueError('Previous attempt may have started; refuse duplicate '+str(old))
 if subprocess.run(['tmux','has-session','-t','dot_pool_e1b_v15_a01'],capture_output=True).returncode==0:raise ValueError('Previous E1b session exists')
 if admission.exists() or run.exists():raise ValueError('Existing execution, refuse duplicate')
 if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Existing session')
 if sha(source/'docs/DOT_pool_pilot_spec_v1.5.md')!=SPEC_SHA:raise ValueError('Spec mismatch')
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
 if not ready['ready'] or ready['spec_sha256']!=PILOT_SPEC_SHA:raise ValueError('Data not ready')
 for sub,key in [('e1','kg_manifest_sha256'),('e2','data_manifest_sha256')]:
  if sha(BASE/'data'/sub/'manifest.json')!=ready[key]:raise ValueError('Manifest changed')
 kg=json.loads((BASE/'data/e1/manifest.json').read_text())
 for name,record in kg['files'].items():
  if sha(BASE/'data/e1'/name)!=record['sha256']:raise ValueError('KG content changed '+name)
 # Only E1 is consumed; E2 remains closed and is not read or retrained.
 pin=json.loads((source/'resources/dot_pool_e1b_v15_admission.json').read_text())
 for name,digest in pin['local_reproduced_analysis_inputs'].items():
  if sha(BASE/'data/e1'/name)!=digest:raise ValueError('Local reconstruction does not match actual node array '+name)
 for path,digest in pin['pilot_routes'].items():
  if sha(path)!=digest:raise ValueError('Pilot route identity differs '+path)
 from scripts.dot_pool_report_recovery import check_burn
 check_burn(pin['burn'])
 for rel,digest in pin['source_files'].items():
  if sha(source/rel)!=digest:raise ValueError('Admission source changed '+rel)
 reference=pilot_reference(source,pin)
 if sha(burn)!=BURN_HASH or sha('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH or Path('/mnt/local/_gpu_guard/DISABLED').exists():raise ValueError('Unexpected burn/guard ownership')
 admission.mkdir();frozen=admission/'source';files=list(source.glob('*.py'))
 for d in ['efficient_llm','scripts','eval','tests']:files+=list((source/d).rglob('*.py'))
 files += [source/p for p in ['resources/llm_pretrain_burn.py','resources/downstream_english_20261023.json','resources/a2_inputs_20261028.json','resources/dot_pool_e1b_v15_admission.json','resources/dot_pool_v12_reference.md.gz','docs/DOT_pool_pilot_spec_v1.5.md','envs/dot_pool.txt']]
 hashes={}
 for path in sorted(set(files)):
  rel=path.relative_to(source);target=frozen/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);hashes[str(rel)]=sha(target)
 (frozen/'pilot_reference.md').write_bytes(reference);hashes['pilot_reference.md']=sha(frozen/'pilot_reference.md')
 write(admission/'source-manifest.json',dict(commit=a.commit,files=hashes))
 write(admission/'data-admission.json',dict(ready=ready,ready_sha256=sha(BASE/'data/ready.json'),physical_content_verified=True,utc=time.time(),runtime={k:version(k) for k in ['torch','transformers','numpy','flash-attn-4','nvidia-cutlass-dsl']}))
 write(admission/'inspection.json',check_burn(pin['burn']))
 # Strict runner JSON deliberately omits study metadata, which lives beside it.
 jobs=job_config(run)
 (admission/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n');load_jobs(admission/'jobs.json')
 env={**os.environ,'DOT_POOL_SPEC_SHA256':E1B_SPEC_SHA,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline','OMP_NUM_THREADS':'2','NCCL_NVLS_ENABLE':'0','PYTHONPATH':str(frozen)}
 argv=[sys.executable,'-u','-m','scripts.train_then_burn','--config',str(admission/'jobs.json'),'--output',str(admission/'supervised'),'--inspection',str(admission/'inspection.json'),'--host',HOST,'--burn-session',session+'_burn','--predecessor-burn-script',str(burn)]
 # tmux servers do not propagate arbitrary client environment keys.
 explicit=['DOT_POOL_SPEC_SHA256','PYTHONPATH','CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','WANDB_MODE','OMP_NUM_THREADS','NCCL_NVLS_ENABLE']
 command='exec '+shlex.join(['env',*[k+'='+env[k] for k in explicit],*argv])+' >'+shlex.quote(str(admission/'supervisor.log'))+' 2>&1'
 subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),command],env=env,check=True)
 subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
 write(admission/'submitted.json',dict(session=session,run=str(run),commit=a.commit,utc=time.time()))
 print('DOT_E1B_V15_SUBMITTED',json.dumps(dict(session=session,admission=str(admission),run=str(run))),flush=True)
if __name__=='__main__':main()
