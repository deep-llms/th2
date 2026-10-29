"""Launch registered Stage1 prerequisites/heals with the unchanged burn supervisor."""
import argparse,json,os,shlex,shutil,socket,subprocess,sys,time
from pathlib import Path
from importlib.metadata import version
from efficient_llm.a2.data import write_json
from efficient_llm.a2_stage1.protocol import SPEC_SHA
from efficient_llm.kv_long import verify_handoff
from scripts.train_then_burn import digest,BURN_HASH,GUARD_HASH
from scripts.gpu_process import inspect
from scripts.launch_a2_downstream import freeze_source
from scripts.a2_prepare_corpora import verify_packed
from run_experiments import load_jobs


def read(p):return json.loads(Path(p).read_text())

def prepare_inputs(source,frozen,original,heal):
 from scripts.a2_preflight_inputs import verify_prepared,_parent_asset_plan
 from scripts.a2_prepare_inputs import verify_inputs
 from scripts.a2_asset_manifest import branch_identity
 plan=read(source/'resources/a2_inputs_20261028.json')
 old=verify_prepared(original,source/'docs/A2_pilot_spec.md',plan)
 for ds,parent in [('dclm','P300'),('dclm','P1T'),('dclm','S72'),('nem','P300')]:verify_inputs(_parent_asset_plan(plan,parent,ds))
 if digest(source/'docs/A2_stage1_spec.md')!=SPEC_SHA:raise ValueError('Stage1 preregistration changed')
 m=read(heal/'manifest.json');complete=read(heal/'complete.json')
 if complete.get('status')!='complete' or complete['manifest_sha256']!=digest(heal/'manifest.json') or not complete['native_reader_passed']:raise ValueError('Shuffled data incomplete')
 if m['schema']!='a2-stage1-shuffled-v1' or m['spec_sha256']!=SPEC_SHA or not m['T7']['passed'] or m['streams']['heal']['rows']<781248:raise ValueError('Shuffled data contract differs')
 if m['source_manifest_sha256']!='9f6ba331da25a6b9763d99f365f613f12e362ea840d4cbfb1ae6027dfb38c7e0':raise ValueError('Wrong source corpus')
 verify_packed(heal,m)
 for name,key in [('documents.bin','documents_sha256'),('document_order.bin','document_order_sha256'),('heldout_documents.bin','heldout_documents_sha256')]:
  if digest(heal/name)!=m[key]:raise ValueError('Document/order identity changed')
 held=Path(m['heldout']['path']);hm=read(held/'manifest.json')
 if digest(held/'manifest.json')!=m['heldout']['manifest_sha256'] or hm['streams']['tuning']['tokens']!=4194304:raise ValueError('Held-out changed')
 verify_packed(held,dict(streams={'tuning':hm['streams']['tuning']}))
 assets=read(original/'assets.json');identity=read(source/'docs/results/a2_review_20261028/identity_metadata.json')['tests']['dclm']
 configs={p:read(Path(assets['dclm/'+p]['weights_dir'])/'config.json') for p in ('P300','P1T','S72')}
 t3=branch_identity(configs['S72'],configs['P1T'],parent='S72',hashes={p:plan['assets']['dclm/'+p]['files']['model.safetensors'] for p in ('P300','P1T','S72','S188')},annealed72000_hash=identity['annealed_step72000_sha256'])
 if assets['dclm/S72']['code_files']!=assets['dclm/P1T']['code_files']:t3=dict(passed=False,reasons=['Branch code differs'])
 history=read(source/'resources/a2_completed_s1_20261028.json');ref=history['sources'][-1]['results']
 if digest(ref['path'])!=ref['sha256']:raise ValueError('T17 historical results changed')
 records=read(ref['path']);records=records if isinstance(records,list) else records['results']
 oldrun=[r for r in records if r['dataset']=='nem' and r['arm']=='P300' and r['budget']=='0.05B' and not r.get('suffix') and r['status']=='complete' and not r['diverged']]
 if len(oldrun)!=1:raise ValueError('Ambiguous T17 reference')
 core={str(p.relative_to(frozen)):digest(p) for p in sorted((frozen/'efficient_llm/a2').glob('*.py'))}
 core.update({str(p.relative_to(frozen)):digest(p) for p in sorted((frozen/'efficient_llm/a2_stage1').glob('*.py'))})
 for name in ('scripts/a2_stage1.py','scripts/launch_a2_stage1.py','packed_evaluation.py','packed_training.py'):core[name]=digest(frozen/name)
 prov=dict(spec_sha256=SPEC_SHA,base_spec_sha256=digest(source/'docs/A2_pilot_spec.md'),data_sha256=digest(heal/'manifest.json'),
  original_data_sha256=old['data_manifest_sha256'],assets_sha256=digest(original/'assets.json'),heldout_sha256=digest(held/'manifest.json'),
  code_commit=os.environ['A2_CODE_COMMIT'],code_files=core,software={k:version(k) for k in ('torch','transformers','accelerate','numpy','safetensors')},microbatch=4,chunk=256)
 return dict(source_plan=str(frozen/'resources/a2_corpora_30b_20261029.json'),original_inputs=str(original),heal=str(heal),heldout=str(held),spec=str(frozen/'docs/A2_stage1_spec.md'),provenance=prov,T3=t3,data_manifest=m,t17_reference=oldrun[0],t17_history_lock=ref,preflight_cap_seconds=7200)



def retire_snapshot():
 """User explicitly retired this exact redundant snapshot; retain its receipts."""
 root=Path('/mnt/local/_data/deep-llms_th2/a2_corpora_30b_20261029/snapshots/dclm_first_3p2b_20261029')
 expected='02ba86c24feb61585f67c710a79bf8bf3cc5a0cf175386c480e26a467075ec87'
 if digest(root/'manifest.json')!=expected:raise ValueError('Snapshot retirement identity changed')
 if (root/'retired.json').is_file():
  if any((root/name).exists() for name in ('heal.bin','heal.starts')):raise ValueError('Partial snapshot retirement')
  return
 manifest=read(root/'manifest.json')['streams']['heal']
 for name,mult,key in [('heal.bin',4,'token_sha256'),('heal.starts',1,'starts_sha256')]:
  f=root/name
  if f.is_symlink() or f.stat().st_size!=3200000000*mult or digest(f)!=manifest[key]:raise ValueError('Snapshot file changed')
 write_json(root/'retirement-authorized.json',dict(manifest_sha256=expected,reason='User requests full30B corpus instead of separate3.2B snapshot',files=['heal.bin','heal.starts']))
 for name in ('heal.bin','heal.starts'):(root/name).unlink()
 (root/'complete.json').rename(root/'original-complete-retired.json')
 write_json(root/'retired.json',dict(manifest_sha256=expected,status='retired',removed_bytes=16000000000,receipts_preserved=True))
 print('SNAPSHOT_RETIRED',str(root),flush=True)


def preflight_jobs(inputs,config,port):
 common=['{python}','-m','accelerate.commands.launch','--config_file',str(config),'--main_process_port',str(port),'-m','efficient_llm.a2_stage1.model_stage']
 jobs=[]
 for parent in ('P300','P1T','S72'):
  jobs.append(dict(name=parent,gpus=list(range(8)),timeout_seconds=1200,argv=common+['parent','--inputs',str(inputs),'--parent',parent,'--output','{run_dir}/'+parent+'.json'],required_outputs=[dict(path=parent+'.json',json_equals={'status':'complete'})]))
 for action,out,seconds,contract in [('numerical','numerical.json',900,{'passed':True}),('throughput','throughput.json',450,{'passed':True}),('t17','t17',1500,{'passed':True})]:
  jobs.append(dict(name=action,gpus=list(range(8)),timeout_seconds=seconds,argv=common+[action,'--inputs',str(inputs),'--output','{run_dir}/'+out,'--estimate','900'],required_outputs=[dict(path=out+('/T17.json' if action=='t17' else ''),json_equals=contract)]))
 jobs.append(dict(name='report',gpus=list(range(8)),timeout_seconds=120,argv=['{python}','-m','scripts.a2_stage1','finalize','--inputs',str(inputs),'--output','{run_dir}'],required_outputs=[dict(path='tests.json',json_equals={'status':'passed'})]))
 return dict(jobs=jobs)


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--stage',choices=('preflight','heals'),required=True)
 for name in ('output','inputs','heal-data','predecessor-burn-script'):p.add_argument('--'+name,type=Path,required=True)
 for name in ('session','code-commit'):p.add_argument('--'+name,required=True)
 p.add_argument('--preflight-root',type=Path);p.add_argument('--preflight-sha256');p.add_argument('--spent-seconds',type=float,default=0)
 a=p.parse_args();source=Path(__file__).resolve().parents[1]
 if socket.gethostname()!='thiennh-p6-6y3o-worker-0' or a.output.exists():raise ValueError('Wrong host/existing output')
 if len(a.code_commit)!=40:raise ValueError('Pinned source required')
 for k,v in [('torch','2.14.1'),('transformers','4.57.1'),('accelerate','1.13.0')]:
  if version(k).split('+')[0]!=v:raise ValueError('Runtime changed')
 for session in (a.session,a.session+'_burn'):
  if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Existing session')
 verify_handoff(a.output.parent,a.predecessor_burn_script,digest)
 if digest(a.predecessor_burn_script)!=BURN_HASH or digest('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH or Path('/mnt/local/_gpu_guard/DISABLED').exists():raise ValueError('Unknown guard/burn owner')
 a.output.mkdir();frozen=freeze_source(source,a.output,a.code_commit)
 extra=['docs/A2_stage1_spec.md','docs/results/a2_review_20261028/identity_metadata.json','resources/a2_completed_s1_20261028.json','resources/a2_corpora_30b_20261029.json']
 sm=read(a.output/'source-manifest.json')
 for name in extra:
  target=frozen/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target);sm['files'][name]=digest(target)
 write_json(a.output/'source-manifest.json',sm);os.environ['A2_CODE_COMMIT']=a.code_commit
 inputs=prepare_inputs(source,frozen,a.inputs,a.heal_data);retire_snapshot();inputpath=a.output/'inputs.json';write_json(inputpath,inputs)
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline','OMP_NUM_THREADS':'1','NCCL_NVLS_ENABLE':'0','TOKENIZERS_PARALLELISM':'false','PYTHONPATH':str(frozen),'A2_CODE_COMMIT':a.code_commit}
 from accelerate.commands.config.config_args import default_config_file,load_config_from_file
 cache=Path(default_config_file);cache.parent.mkdir(parents=True,exist_ok=True)
 if cache.exists():shutil.copyfile(cache,a.output/'previous-accelerate.yaml')
 config=frozen/'resources/accelerate_b200.yaml';shutil.copyfile(config,cache);actual=load_config_from_file(str(cache))
 if cache.read_bytes()!=config.read_bytes() or actual.num_processes!=8 or actual.mixed_precision!='bf16' or actual.use_cpu:raise ValueError('Accelerate mismatch')
 receipt=subprocess.run([sys.executable,'-m','accelerate.commands.env'],check=True,capture_output=True,text=True,env={**env,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
 (a.output/'accelerate-env.log').write_text(receipt.stdout+receipt.stderr);print(receipt.stdout,flush=True)
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 if a.stage=='preflight':jobs=preflight_jobs(inputpath,config,port)
 else:
  if not a.preflight_sha256 or digest(a.preflight_root/'supervised/run/tests.json')!=a.preflight_sha256:raise ValueError('Reviewed preflight report SHA required before heals')
  old=read(a.preflight_root/'inputs.json');gate=read(a.preflight_root/'supervised/run/tests.json')
  # Source freeze path/code commit may differ, numerical source/runtime/data may not.
  for key in ('code_files','spec_sha256','data_sha256','original_data_sha256','assets_sha256','heldout_sha256','software','microbatch','chunk'):
   if old['provenance'][key]!=inputs['provenance'][key]:raise ValueError('Preflight evidence differs: '+key)
  if gate['report_before_launch']:raise ValueError('Report >30h budget before healing')
  inputs['provenance']=old['provenance'];write_json(inputpath,inputs)
  timeout=min(36*3600-a.spent_seconds-600,math_ceil(gate['estimated_remaining_seconds']*1.5))
  if timeout<=0:raise ValueError('No remaining budget')
  jobs=dict(jobs=[dict(name='stage1-heals',gpus=list(range(8)),timeout_seconds=timeout,argv=['{python}','-m','scripts.a2_stage1','heals','--inputs',str(inputpath),'--preflight',str(a.preflight_root/'supervised/run'),'--output','{run_dir}/results/a2_stage1','--port',str(port),'--spent-seconds',str(a.spent_seconds)],required_outputs=[dict(path='results/a2_stage1/complete.json',json_equals={'status':'complete','training_complete':True})])])
 jobfile=a.output/'jobs.json';write_json(jobfile,jobs);load_jobs(jobfile);write_json(a.output/'inspection.json',inspect())
 keys=('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','WANDB_MODE','OMP_NUM_THREADS','NCCL_NVLS_ENABLE','TOKENIZERS_PARALLELISM','PYTHONPATH','A2_CODE_COMMIT')
 argv=['env',*[k+'='+env[k] for k in keys],sys.executable,'-u','-m','scripts.train_then_burn','--config',str(jobfile),'--output',str(a.output/'supervised'),'--inspection',str(a.output/'inspection.json'),'--host','thiennh-p6-6y3o-worker-0','--burn-session',a.session+'_burn','--predecessor-burn-script',str(a.predecessor_burn_script)]
 command='exec '+shlex.join(argv)+' >'+shlex.quote(str(a.output/'supervisor.log'))+' 2>&1'
 subprocess.run(['tmux','new-session','-d','-s',a.session,'-c',str(frozen),command],env=env,check=True)
 subprocess.run(['tmux','set-option','-w','-t',a.session,'remain-on-exit','on'],check=True)
 print('STAGE1_SUBMITTED',a.stage,a.output,flush=True)

def math_ceil(x):
 import math
 return math.ceil(x)
if __name__=='__main__':main()
