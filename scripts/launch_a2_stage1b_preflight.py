"""Launch registered Stage1b preflight with the unchanged burn supervisor."""
import argparse,json,os,shlex,shutil,socket,subprocess,sys,time
from pathlib import Path
from importlib.metadata import version
from efficient_llm.a2.data import write_json
from efficient_llm.a2_stage1b.protocol import SPEC_SHA, REPLICATION_SPEC_SHA
from efficient_llm.kv_long import verify_handoff
from scripts.train_then_burn import digest,BURN_HASH,GUARD_HASH
from scripts.gpu_process import inspect
from scripts.launch_a2_downstream import freeze_source
from scripts.a2_prepare_corpora import verify_packed
from run_experiments import load_jobs
from scripts.a2_stage1b_budget import budget


def read(p):return json.loads(Path(p).read_text())

def prepare_inputs(source,frozen,original,heal):
 from scripts.a2_preflight_inputs import verify_prepared,_parent_asset_plan
 from scripts.a2_prepare_inputs import verify_inputs
 from scripts.a2_asset_manifest import branch_identity
 plan=read(source/'resources/a2_inputs_20261028.json')
 old=verify_prepared(original,source/'docs/A2_pilot_spec.md',plan)
 for ds,parent in [('fwe','P300'),('fwe','P1T'),('fwe','S72'),('nem','P300')]:verify_inputs(_parent_asset_plan(plan,parent,ds))
 if digest(source/'docs/A2_stage1_spec.md')!=SPEC_SHA:raise ValueError('Stage1 preregistration changed')
 m=read(heal/'manifest.json');complete=read(heal/'complete.json')
 if complete.get('status')!='complete' or complete['manifest_sha256']!=digest(heal/'manifest.json') or not complete['native_reader_passed']:raise ValueError('Shuffled data incomplete')
 if m.get('stage1b_spec_sha256')!=REPLICATION_SPEC_SHA or m.get('corpus')!='fineweb_edu':raise ValueError('Wrong replication data')
 if m['schema']!='a2-stage1-shuffled-v1' or m['spec_sha256']!=SPEC_SHA or not m['T7']['passed'] or m['streams']['heal']['rows']<781248:raise ValueError('Shuffled data contract differs')
 if m['source_manifest_sha256']!='0685a790f3c60bf0c02f2e0679910ada71d9c3fefd95a99c53578b201b87f3dd':raise ValueError('Wrong source corpus')
 verify_packed(heal,m)
 for name,key in [('documents.bin','documents_sha256'),('document_order.bin','document_order_sha256'),('heldout_documents.bin','heldout_documents_sha256')]:
  if digest(heal/name)!=m[key]:raise ValueError('Document/order identity changed')
 held=Path(m['heldout']['path']);hm=read(held/'manifest.json')
 if digest(held/'manifest.json')!=m['heldout']['manifest_sha256'] or hm['streams']['tuning']['tokens']!=4194304:raise ValueError('Held-out changed')
 verify_packed(held,dict(streams={'tuning':hm['streams']['tuning']}))
 assets=read(original/'assets.json');identity=read(source/'docs/results/a2_review_20261028/identity_metadata.json')['tests']['fwe']
 configs={p:read(Path(assets['fwe/'+p]['weights_dir'])/'config.json') for p in ('P300','P1T','S72')}
 t3=branch_identity(configs['S72'],configs['P1T'],parent='S72',hashes={p:plan['assets']['fwe/'+p]['files']['model.safetensors'] for p in ('P300','P1T','S72','S188')},annealed72000_hash=identity['annealed_step72000_sha256'])
 if assets['fwe/S72']['code_files']!=assets['fwe/P1T']['code_files']:t3=dict(passed=False,reasons=['Branch code differs'])
 history=read(source/'resources/a2_completed_s1_20261028.json');ref=history['sources'][-1]['results']
 if digest(ref['path'])!=ref['sha256']:raise ValueError('T17 historical results changed')
 records=read(ref['path']);records=records if isinstance(records,list) else records['results']
 oldrun=[r for r in records if r['dataset']=='nem' and r['arm']=='P300' and r['budget']=='0.05B' and not r.get('suffix') and r['status']=='complete' and not r['diverged']]
 if len(oldrun)!=1:raise ValueError('Ambiguous T17 reference')
 core={str(p.relative_to(frozen)):digest(p) for p in sorted((frozen/'efficient_llm/a2').glob('*.py'))}
 core.update({str(p.relative_to(frozen)):digest(p) for p in sorted((frozen/'efficient_llm/a2_stage1').glob('*.py'))})
 core.update({str(p.relative_to(frozen)):digest(p) for p in sorted((frozen/'efficient_llm/a2_stage1b').glob('*.py'))})
 for name in ('scripts/a2_stage1.py','scripts/launch_a2_stage1.py','scripts/a2_stage1b.py','scripts/a2_stage1b_preflight.py','scripts/launch_a2_stage1b_preflight.py','packed_evaluation.py','packed_training.py'):core[name]=digest(frozen/name)
 prov=dict(spec_sha256=SPEC_SHA,stage1b_spec_sha256=REPLICATION_SPEC_SHA,dataset="fwe",base_spec_sha256=digest(source/'docs/A2_pilot_spec.md'),data_sha256=digest(heal/'manifest.json'),
  original_data_sha256=old['data_manifest_sha256'],assets_sha256=digest(original/'assets.json'),heldout_sha256=digest(held/'manifest.json'),
  code_commit=os.environ['A2_CODE_COMMIT'],code_files=core,software={k:version(k) for k in ('torch','transformers','accelerate','numpy','safetensors')},microbatch=4,chunk=256)
 return dict(dataset='fwe',replication_spec=str(frozen/'docs/A2_stage1b_fwe_spec.md'),source_plan=str(frozen/'resources/a2_corpora_30b_20261029.json'),original_inputs=str(original),heal=str(heal),heldout=str(held),spec=str(frozen/'docs/A2_stage1_spec.md'),provenance=prov,T3=t3,data_manifest=m,t17_reference=oldrun[0],t17_history_lock=ref,preflight_cap_seconds=7200)


def preflight_jobs(inputs,config,port):
 common=['{python}','-m','accelerate.commands.launch','--config_file',str(config),'--main_process_port',str(port),'-m','efficient_llm.a2_stage1b.model_stage']
 jobs=[]
 for parent in ('P300','P1T','S72'):
  jobs.append(dict(name=parent,gpus=list(range(8)),timeout_seconds=1200,argv=common+['parent','--inputs',str(inputs),'--parent',parent,'--output','{run_dir}/'+parent+'.json'],required_outputs=[dict(path=parent+'.json',json_equals={'status':'complete'})]))
 for action,out,seconds,contract in [('numerical','numerical.json',900,{'passed':True}),('throughput','throughput.json',450,{'passed':True}),('t17','t17',1500,{'passed':True})]:
  jobs.append(dict(name=action,gpus=list(range(8)),timeout_seconds=seconds,argv=common+[action,'--inputs',str(inputs),'--output','{run_dir}/'+out,'--estimate','900'],required_outputs=[dict(path=out+('/T17.json' if action=='t17' else ''),json_equals=contract)]))
 jobs.append(dict(name='report',gpus=list(range(8)),timeout_seconds=120,argv=['{python}','-m','scripts.a2_stage1b_preflight','--inputs',str(inputs),'--output','{run_dir}'],required_outputs=[dict(path='tests.json',json_equals={'status':'passed'})]))
 return dict(jobs=jobs)


def main():
 p=argparse.ArgumentParser()
 for name in ('output','inputs','heal-data','predecessor-burn-script'):p.add_argument('--'+name,type=Path,required=True)
 for name in ('session','code-commit'):p.add_argument('--'+name,required=True)
 p.add_argument('--deadline',required=True);p.add_argument('--spent-seconds',type=float,required=True)
 a=p.parse_args();started=time.monotonic();source=Path(__file__).resolve().parents[1]
 budget(a.spent_seconds,0,7200,a.deadline)
 if socket.gethostname()!='thiennh-p6-6y3o-worker-0' or a.output.exists():raise ValueError('Wrong host/existing output')
 if len(a.code_commit)!=40:raise ValueError('Pinned source required')
 for k,v in [('torch','2.14.1'),('transformers','4.57.1'),('accelerate','1.13.0')]:
  if version(k).split('+')[0]!=v:raise ValueError('Runtime changed')
 for session in (a.session,a.session+'_burn'):
  if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Existing session')
 verify_handoff(a.output.parent,a.predecessor_burn_script,digest)
 if digest(a.predecessor_burn_script)!=BURN_HASH or digest('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH or Path('/mnt/local/_gpu_guard/DISABLED').exists():raise ValueError('Unknown guard/burn owner')
 if digest(source/'docs/A2_stage1b_fwe_spec.md')!=REPLICATION_SPEC_SHA:raise ValueError('Replication spec changed')
 a.output.mkdir();frozen=freeze_source(source,a.output,a.code_commit)
 extra=['docs/A2_stage1b_fwe_spec.md','resources/a2_stage1b_fwe_release_20261030.json','docs/A2_stage1_spec.md','docs/results/a2_review_20261028/identity_metadata.json','resources/a2_completed_s1_20261028.json','resources/a2_corpora_30b_20261029.json']
 sm=read(a.output/'source-manifest.json')
 for name in extra:
  target=frozen/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target);sm['files'][name]=digest(target)
 write_json(a.output/'source-manifest.json',sm);os.environ['A2_CODE_COMMIT']=a.code_commit
 inputs=prepare_inputs(source,frozen,a.inputs,a.heal_data);inputpath=a.output/'inputs.json';write_json(inputpath,inputs)
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline','OMP_NUM_THREADS':'1','NCCL_NVLS_ENABLE':'0','TOKENIZERS_PARALLELISM':'false','PYTHONPATH':str(frozen),'A2_CODE_COMMIT':a.code_commit}
 from accelerate.commands.config.config_args import default_config_file,load_config_from_file
 cache=Path(default_config_file);cache.parent.mkdir(parents=True,exist_ok=True)
 if cache.exists():shutil.copyfile(cache,a.output/'previous-accelerate.yaml')
 config=frozen/'resources/accelerate_b200.yaml';shutil.copyfile(config,cache);actual=load_config_from_file(str(cache))
 if cache.read_bytes()!=config.read_bytes() or actual.num_processes!=8 or actual.mixed_precision!='bf16' or actual.use_cpu:raise ValueError('Accelerate mismatch')
 receipt=subprocess.run([sys.executable,'-m','accelerate.commands.env'],check=True,capture_output=True,text=True,env={**env,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
 (a.output/'accelerate-env.log').write_text(receipt.stdout+receipt.stderr);print(receipt.stdout,flush=True)
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 jobs=preflight_jobs(inputpath,config,port)
 write_json(a.output/'stage-budget.json',budget(a.spent_seconds,time.monotonic()-started,7200,a.deadline))
 jobfile=a.output/'jobs.json';write_json(jobfile,jobs);load_jobs(jobfile);write_json(a.output/'inspection.json',inspect())
 keys=('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','WANDB_MODE','OMP_NUM_THREADS','NCCL_NVLS_ENABLE','TOKENIZERS_PARALLELISM','PYTHONPATH','A2_CODE_COMMIT')
 argv=['env',*[k+'='+env[k] for k in keys],sys.executable,'-u','-m','scripts.train_then_burn','--config',str(jobfile),'--output',str(a.output/'supervised'),'--inspection',str(a.output/'inspection.json'),'--host','thiennh-p6-6y3o-worker-0','--burn-session',a.session+'_burn','--predecessor-burn-script',str(a.predecessor_burn_script)]
 command='exec '+shlex.join(argv)+' >'+shlex.quote(str(a.output/'supervisor.log'))+' 2>&1'
 subprocess.run(['tmux','new-session','-d','-s',a.session,'-c',str(frozen),command],env=env,check=True)
 subprocess.run(['tmux','set-option','-w','-t',a.session,'remain-on-exit','on'],check=True)
 print('STAGE1B_PREFLIGHT_SUBMITTED',a.output,flush=True)

def math_ceil(x):
 import math
 return math.ceil(x)
if __name__=='__main__':main()
