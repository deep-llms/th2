"""Launch one registered FWE part after reviewed preflight and native handoff."""
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


from scripts.launch_a2_stage1b_preflight import read, prepare_inputs
from efficient_llm.a2_stage1b.protocol import phase_runs
from scripts.a2_remaining import read_locked
import math


def admit_part2(prior_lock, predecessor):
    prior=read_locked(prior_lock)
    if prior.get('stage1b_spec_sha256')!=REPLICATION_SPEC_SHA:
        raise ValueError('Prior Stage1b specification mismatch')
    interim=read_locked(prior['interim'])
    if interim.get('status')!='verified' or interim.get('report_sent') is not True:
        raise ValueError('Send verified interim before Part2')
    for key in ('report','decision'):
        item=interim[key]
        if digest(item['path'])!=item['sha256']:raise ValueError('Interim artifact changed')
    root=predecessor.resolve().parents[2]
    for key,rel in [('supervisor','supervised/supervisor.json'),
                    ('run_complete','supervised/run/complete.json'),
                    ('burn','supervised/burn-verified.json'),
                    ('complete','supervised/run/downstream/complete.json')]:
        item=prior['downstream'][key]
        if Path(item['path']).resolve()!=root/rel:
            raise ValueError('Part1 downstream predecessor mismatch')
        obj=read_locked(item)
        if key=='complete' and (obj.get('status')!='ok' or obj.get('models')!=4):
            raise ValueError('Four-model downstream incomplete')
    return prior


def phase_estimate(gate, phase):
    if phase == 'part1':
        steps=8*762+2*12207; fits=10
    else:
        runs=phase_runs('part2',.001,gate['eligible_x72'])
        steps=sum(r.steps for r in runs);fits=len(runs)
    return steps*gate['step_seconds']+fits*180


def heals_jobs(inputs,preflight,port,phase,spent,timeout,prior=None,prior_sha=None):
    argv=['{python}','-m','scripts.a2_stage1b','--phase',phase,'--inputs',str(inputs),
          '--preflight',str(preflight),'--output','{run_dir}/results/a2_stage1_fwe',
          '--port',str(port),'--spent-seconds',str(spent)]
    if prior:argv+=['--prior',prior,'--prior-sha256',prior_sha]
    return dict(jobs=[dict(name='stage1b-'+phase,gpus=list(range(8)),timeout_seconds=timeout,
        argv=argv,required_outputs=[dict(path='results/a2_stage1_fwe/complete.json',
        json_equals={'status':'complete','phase':phase,'training_complete':phase=='part2'})])])


def main():
 p=argparse.ArgumentParser()
 for name in ('output','inputs','heal-data','predecessor-burn-script'):p.add_argument('--'+name,type=Path,required=True)
 for name in ('session','code-commit'):p.add_argument('--'+name,required=True)
 p.add_argument('--deadline',required=True);p.add_argument('--spent-seconds',type=float,required=True)
 p.add_argument('--phase',choices=('part1','part2'),required=True)
 p.add_argument('--preflight-root',type=Path,required=True);p.add_argument('--preflight-sha256',required=True)
 p.add_argument('--prior');p.add_argument('--prior-sha256')
 a=p.parse_args();started=time.monotonic();source=Path(__file__).resolve().parents[1]
 gate=read(a.preflight_root/'supervised/run/tests.json')
 if digest(a.preflight_root/'supervised/run/tests.json')!=a.preflight_sha256 or gate['status']!='passed':raise ValueError('Reviewed passing preflight SHA required')
 if gate['report_before_launch']:raise ValueError('Report over20h estimate before launch')
 if a.phase=='part1' and (a.prior or a.prior_sha256):raise ValueError('Unexpected prior part1')
 if a.phase=='part2' and not (a.prior and a.prior_sha256):raise ValueError('Verified interim lock required')
 if a.phase=='part2':admit_part2(dict(path=a.prior,sha256=a.prior_sha256),a.predecessor_burn_script)
 if a.phase=='part1' and a.predecessor_burn_script.resolve().parents[2]!=a.preflight_root.resolve():raise ValueError('Preflight predecessor mismatch')
 remaining=phase_estimate(gate,a.phase)
 timeout=math.ceil(remaining*1.5)
 receipt=budget(a.spent_seconds,0,timeout,a.deadline)
 if receipt['report_due']:raise ValueError('Report over20h admission before launch')
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
 inputs=prepare_inputs(source,frozen,a.inputs,a.heal_data)
 old=read(a.preflight_root/'inputs.json')
 for key in ('code_files','spec_sha256','stage1b_spec_sha256','dataset','data_sha256','original_data_sha256','assets_sha256','heldout_sha256','software','microbatch','chunk'):
  if old['provenance'][key]!=inputs['provenance'][key]:raise ValueError('Preflight evidence differs: '+key)
 if gate['provenance']!=old['provenance']:raise ValueError('Preflight provenance mismatch')
 inputs['provenance']=old['provenance'];inputpath=a.output/'inputs.json';write_json(inputpath,inputs)
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline','OMP_NUM_THREADS':'1','NCCL_NVLS_ENABLE':'0','TOKENIZERS_PARALLELISM':'false','PYTHONPATH':str(frozen),'A2_CODE_COMMIT':a.code_commit}
 from accelerate.commands.config.config_args import default_config_file,load_config_from_file
 cache=Path(default_config_file);cache.parent.mkdir(parents=True,exist_ok=True)
 if cache.exists():shutil.copyfile(cache,a.output/'previous-accelerate.yaml')
 config=frozen/'resources/accelerate_b200.yaml';shutil.copyfile(config,cache);actual=load_config_from_file(str(cache))
 if cache.read_bytes()!=config.read_bytes() or actual.num_processes!=8 or actual.mixed_precision!='bf16' or actual.use_cpu:raise ValueError('Accelerate mismatch')
 receipt=subprocess.run([sys.executable,'-m','accelerate.commands.env'],check=True,capture_output=True,text=True,env={**env,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
 (a.output/'accelerate-env.log').write_text(receipt.stdout+receipt.stderr);print(receipt.stdout,flush=True)
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 jobs=heals_jobs(inputpath,a.preflight_root/'supervised/run',port,a.phase,a.spent_seconds+time.monotonic()-started,timeout,a.prior,a.prior_sha256)
 write_json(a.output/'stage-budget.json',budget(a.spent_seconds,time.monotonic()-started,timeout,a.deadline))
 jobfile=a.output/'jobs.json';write_json(jobfile,jobs);load_jobs(jobfile);write_json(a.output/'inspection.json',inspect())
 keys=('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','WANDB_MODE','OMP_NUM_THREADS','NCCL_NVLS_ENABLE','TOKENIZERS_PARALLELISM','PYTHONPATH','A2_CODE_COMMIT')
 argv=['env',*[k+'='+env[k] for k in keys],sys.executable,'-u','-m','scripts.train_then_burn','--config',str(jobfile),'--output',str(a.output/'supervised'),'--inspection',str(a.output/'inspection.json'),'--host','thiennh-p6-6y3o-worker-0','--burn-session',a.session+'_burn','--predecessor-burn-script',str(a.predecessor_burn_script)]
 command='exec '+shlex.join(argv)+' >'+shlex.quote(str(a.output/'supervisor.log'))+' 2>&1'
 subprocess.run(['tmux','new-session','-d','-s',a.session,'-c',str(frozen),command],env=env,check=True)
 subprocess.run(['tmux','set-option','-w','-t',a.session,'remain-on-exit','on'],check=True)
 print('STAGE1B_HEALS_SUBMITTED',a.output,flush=True)

if __name__=='__main__':main()
