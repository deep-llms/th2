"""Why launch through the unchanged native supervisor with exact predecessor admission."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
from efficient_llm.a2_why.common import read,write,Checks,SPEC_SHA
from scripts.a2_quant_io import lock,write as native_write
from scripts.a2_remaining import read_locked
from scripts.launch_a2_downstream import freeze_source
from scripts.launch_a2_depth_partb import predecessor
from scripts.train_then_burn import digest
from run_experiments import load_jobs


def main():
    p=argparse.ArgumentParser()
    for name in ('plan','output','session','code-commit'):p.add_argument('--'+name,required=True)
    p.add_argument('--stage',choices=('yt1','rest'),required=True);a=p.parse_args()
    source=Path(__file__).resolve().parents[1];cfg=read(a.plan);out=Path(a.output)
    if socket.gethostname()!='thiennh-p6-wc69-worker-0' or out.exists():raise ValueError('Wrong node or existing root')
    for session in (a.session,a.session+'_burn'):
        if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Duplicate session')
    admission=predecessor(cfg,out)
    out.mkdir();rel=str(Path(a.plan).resolve().relative_to(source))
    frozen=freeze_source(source,out,a.code_commit,extra_files=(rel,'docs/A2_why_spec.md','resources/a2_why_cpu_20261104.json','resources/a2_depth_stage1_references_20261101.json'))
    root=out/'results/a2_why_20261104';root.mkdir(parents=True)
    checks=Checks(root/'preparation_checks.json')
    checks.record('Why_spec',cfg['spec_sha256']==SPEC_SHA==digest(frozen/'docs/A2_why_spec.md'))
    proof=read(frozen/'resources/a2_why_cpu_20261104.json')
    checks.record('CPU_proof',proof['status']=='PASS' and all(digest(frozen/n)==h for n,h in proof['files'].items()),proof)
    from efficient_llm.a2.data import PackedRows
    for label in ('tuning','final'):checks.attempt('data:'+label,lambda label=label:PackedRows(cfg['data_root'],label,verify=True))
    checks.attempt('assets',lambda:read_locked(cfg['assets']))
    checks.attempt('data_manifest',lambda:read_locked(cfg['data_manifest']))
    for key,item in cfg['reference_nll'].items():
        if a.stage=='rest' or key.startswith('dclm/P300/'):
            checks.attempt('reference:'+key,lambda item=item:read_locked(item))
    if a.stage=='rest':
        data=read_locked(cfg['why_data']);cfg['heal_streams']=data['streams']
        checks.record('Why_data_complete',data['status']=='complete',cfg['why_data'])
    for ds,streams in cfg['heal_streams'].items():
        for seed,item in streams.items():
            def verify(item=item):
                m=read_locked(dict(path=item['path']+'/manifest.json',sha256=item['sha256']))
                # New streams record YT2 flags; propagate them to every affected run.
                for c in m.get('checks',[]):checks.record('stream:'+ds+':'+seed+':'+c['test'],c['passed'],c.get('detail'))
                return m
            checks.attempt('stream_manifest:'+ds+':'+seed,verify)
    cfg.update(results_root=str(root),code_commit=a.code_commit,source_manifest=lock(out/'source-manifest.json'),
        preparation_flags=[c for c in checks.items if not c['passed']])
    config=write(root/'config/plan.json',cfg);shutil.copyfile(frozen/'docs/A2_why_spec.md',root/'config/spec.md')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NCCL_NVLS_ENABLE':'0','TOKENIZERS_PARALLELISM':'false','PYTHONPATH':str(frozen),'A2_CODE_COMMIT':a.code_commit}
    from accelerate.commands.config.config_args import default_config_file,load_config_from_file
    cache=Path(default_config_file);cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():shutil.copyfile(cache,out/'previous-accelerate.yaml')
    expected=frozen/'resources/accelerate_b200.yaml';shutil.copyfile(expected,cache);actual=load_config_from_file(str(cache))
    checks.record('accelerate_config',cache.read_bytes()==expected.read_bytes() and actual.num_processes==8 and actual.mixed_precision=='bf16' and not actual.use_cpu)
    test=subprocess.run([sys.executable,'-m','accelerate.commands.env'],capture_output=True,text=True,env={**env,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
    (out/'accelerate-env.log').write_text(test.stdout+test.stderr);checks.record('accelerate_env',test.returncode==0)
    admission=predecessor(cfg,out);write(out/'admission.json',admission);write(out/'inspection.json',admission['inspection'])
    jobs=native_write(out/'jobs.json',dict(jobs=[dict(name='a2-why-'+a.stage,gpus=list(range(8)),timeout_seconds=None,
        argv=['{python}','-u','-m','scripts.a2_why_run','--config',config['path'],'--stage',a.stage,'--receipt','{run_dir}/why-complete.json'],
        required_outputs=[dict(path='why-complete.json',json_equals=dict(execution_complete=True))])]))
    load_jobs(Path(jobs['path']))
    argv=[sys.executable,'-u','-m','scripts.train_then_burn','--config',jobs['path'],'--output',str(out/'supervised'),'--inspection',str(out/'inspection.json'),
        '--host',socket.gethostname(),'--burn-session',a.session+'_burn','--predecessor-burn-script',cfg['predecessor']['burn_script']]
    command='exec '+shlex.join(argv)+' >'+shlex.quote(str(out/'supervisor.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',a.session,'-c',str(frozen),command],env=env,check=True)
    subprocess.run(['tmux','set-option','-w','-t',a.session,'remain-on-exit','on'],check=True)
    print('WHY_SUBMITTED',a.stage,str(out),flush=True)

if __name__=='__main__':main()
