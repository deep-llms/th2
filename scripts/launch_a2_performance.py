"""User-authorized discarded A2 optimization probes; existing supervisor unchanged."""
import argparse,json,os,shlex,shutil,socket,subprocess,sys
from pathlib import Path
from importlib.metadata import version
from efficient_llm.kv_long import verify_handoff
from efficient_llm.a2.data import write_json
from efficient_llm.a2.performance import SUITES
from scripts.train_then_burn import digest,BURN_HASH,GUARD_HASH
from scripts.gpu_process import inspect
from scripts.launch_a2_downstream import freeze_source
from run_experiments import load_jobs


def jobs_for(inputs,config,port,suite="initial"):
    variants=SUITES[suite]
    launch=['{python}','-m','accelerate.commands.launch','--config_file',str(config),
            '--main_process_port',str(port),'-m','efficient_llm.a2.perf_probe','--inputs',str(inputs),'--suite',suite]
    jobs=[]
    for name in variants:
        args=launch+['--variant',name,'--output','{run_dir}/'+name]
        if name=='reference':
            if suite=='initial':args+=['--profile']
        else:args+=['--reference','{run_dir}/reference']
        jobs.append(dict(name=name,gpus=list(range(8)),timeout_seconds=900 if name=='compiled_head' else 450,
            argv=args,required_outputs=[dict(path=name+'/result.json',json_equals=dict(status='complete'))]))
    # Numerical failure is recorded, not accepted. A process failure aborts the
    # native queue and restores burns; no ad-hoc cleanup or retry in this probe.
    return dict(jobs=jobs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','inputs','predecessor-burn-script'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('host','session','code-commit'):p.add_argument('--'+name,required=True)
    p.add_argument('--suite',choices=SUITES,default='initial')
    a=p.parse_args();source=Path(__file__).resolve().parents[1]
    if socket.gethostname()!=a.host or a.output.exists():raise ValueError('Wrong host/existing output')
    if len(a.code_commit)!=40 or any(c not in '0123456789abcdef' for c in a.code_commit):raise ValueError('Pinned source required')
    for name in (a.session,a.session+'_burn'):
        if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0:raise ValueError('Existing session')
    for k,v in [('torch','2.14.1'),('transformers','4.57.1'),('accelerate','1.13.0')]:
        if version(k).split('+')[0]!=v:raise ValueError('Runtime changed')
    verify_handoff(a.output.parent,a.predecessor_burn_script,digest)
    if digest(a.predecessor_burn_script)!=BURN_HASH or digest('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH or Path('/mnt/local/_gpu_guard/DISABLED').exists():
        raise ValueError('Unknown guard/burn owner')
    from scripts.a2_preflight_inputs import verify_prepared,initial_asset_plan
    from scripts.a2_prepare_inputs import verify_inputs
    plan=json.loads((source/'resources/a2_inputs_20261028.json').read_text())
    data=verify_prepared(a.inputs,source/'docs/A2_pilot_spec.md',plan)
    weights=verify_inputs(initial_asset_plan(plan))
    a.output.mkdir();frozen=freeze_source(source,a.output,a.code_commit)
    write_json(a.output/'inputs-verified.json',dict(data=data,weights=weights))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline',
        'OMP_NUM_THREADS':'1','NCCL_NVLS_ENABLE':'0','TOKENIZERS_PARALLELISM':'false',
        'PYTHONPATH':str(frozen),'A2_CODE_COMMIT':a.code_commit,
        'TORCHINDUCTOR_CACHE_DIR':str(a.output/'inductor-cache'),
        'TRITON_CACHE_DIR':str(a.output/'triton-cache'),'TORCHINDUCTOR_COMPILE_THREADS':'1'}
    from accelerate.commands.config.config_args import default_config_file,load_config_from_file
    cache=Path(default_config_file);cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():shutil.copyfile(cache,a.output/'previous-accelerate.yaml')
    config=frozen/'resources/accelerate_b200.yaml';shutil.copyfile(config,cache)
    actual=load_config_from_file(str(cache))
    if cache.read_bytes()!=config.read_bytes() or actual.num_processes!=8 or actual.mixed_precision!='bf16' or actual.use_cpu:raise ValueError('Accelerate mismatch')
    receipt=subprocess.run([sys.executable,'-m','accelerate.commands.env'],check=True,capture_output=True,text=True,
        env={**env,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
    (a.output/'accelerate-env.log').write_text(receipt.stdout+receipt.stderr)
    print(receipt.stdout,flush=True)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    jobs=jobs_for(a.inputs,config,port,a.suite)
    if sum(j['timeout_seconds'] for j in jobs['jobs'])>5000:raise ValueError('Probe budget exceeds bound')
    write_json(a.output/'plan.json',dict(suite=a.suite,scope='discarded_performance_only',steps_per_variant=16,
        global_tokens_per_step=262144,variants={k:v.record() for k,v in SUITES[a.suite].items()},
        maximum_job_seconds=sum(j['timeout_seconds'] for j in jobs['jobs']),
        compile_scope='head/loss only, fullgraph Inductor, backward autocast off' if a.suite=='initial' else 'none; unchanged native KDModel/HealTrainer',
        thresholds=dict(gradient_relative_l2=.02,gradient_cosine=.9998,validation_abs_nats=.002)))
    jobfile=a.output/'jobs.json';write_json(jobfile,jobs);load_jobs(jobfile)
    write_json(a.output/'inspection.json',inspect())
    keys=('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE',
        'HF_HUB_DISABLE_TELEMETRY','WANDB_MODE','OMP_NUM_THREADS','NCCL_NVLS_ENABLE','TOKENIZERS_PARALLELISM',
        'PYTHONPATH','A2_CODE_COMMIT','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_COMPILE_THREADS')
    argv=['env',*[k+'='+env[k] for k in keys],sys.executable,'-u','-m','scripts.train_then_burn',
        '--config',str(jobfile),'--output',str(a.output/'supervised'),'--inspection',str(a.output/'inspection.json'),
        '--host',a.host,'--burn-session',a.session+'_burn','--predecessor-burn-script',str(a.predecessor_burn_script)]
    command='exec '+shlex.join(argv)+' >'+shlex.quote(str(a.output/'supervisor.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',a.session,'-c',str(frozen),command],env=env,check=True)
    subprocess.run(['tmux','set-option','-w','-t',a.session,'remain-on-exit','on'],check=True)
    print('A2_PERFORMANCE_SUBMITTED',a.output,flush=True)


if __name__=='__main__':main()
