"""Launch registered C through the existing native ownership supervisor."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
from efficient_llm.a2_depth.common import Checks,read,write
from scripts.a2_quant_io import lock
from scripts.a2_remaining import read_locked
from scripts.launch_a2_downstream import freeze_source
from scripts.launch_a2_depth_partb import predecessor
from scripts.train_then_burn import digest


def job_config(config):
    return dict(jobs=[dict(name='a2-depth-partc',gpus=list(range(8)),timeout_seconds=None,
        argv=['{python}','-u','-m','scripts.a2_depth_partc_run','--config',config['path'],
              '--receipt','{run_dir}/depth-partc-complete.json'],
        required_outputs=[dict(path='depth-partc-complete.json',json_equals=dict(execution_complete=True))])])


def prepare(cfg,root,frozen,commit):
    """All scientific failures are recorded; physical admission is outside this function."""
    checks=Checks(root/'preparation_checks.json')
    checks.record('spec_identity',digest(frozen/'docs/A2_depth_spec.md')==cfg['spec_sha256'])
    from efficient_llm.a2.data import PackedRows
    for label in ('tuning','final'):
        checks.attempt('packed_data:'+label,lambda label=label:PackedRows(cfg['data_root'],label,verify=True))
    for name in ('assets','data_manifest','prepared'):
        checks.attempt(name,lambda name=name:read_locked(cfg[name]))
    for name,item in cfg['reference_nll'].items():
        checks.attempt('reference:'+name,lambda item=item:read_locked(item))
    actual=checks.attempt('B_terminal',lambda:read_locked(cfg['predecessor']['complete']))
    checks.record('B_coverage',bool(actual and actual.get('coverage_complete') and actual.get('readout_complete')),actual)
    note=cfg['part_ab_note'];checks.record('A_B_report_posted',bool(note.get('posted_at')) and note.get('spec_sha256')==cfg['spec_sha256'],note)
    def copy_report():
        item=cfg['part_ab_report'];text=Path(item['path']).read_text()
        checks.record('A_B_report_identity',digest(item['path'])==item['sha256'],item)
        (root/'PART_AB_REPORT.md').write_text(text);cfg['part_ab_report_local']=str(root/'PART_AB_REPORT.md')
    checks.attempt('copy_A_B_report',copy_report)
    def historical():
        history=cfg['stream_history'];original=read_locked(history['historical_inputs'])
        checks.record('DT6_historical_manifest',original['data_manifest']==history['data_manifest'])
        checks.record('DT6_historical_provenance',original['provenance']==history['provenance'])
    checks.attempt('DT6_original_receipt',historical)
    def cpu():
        evidence=read(frozen/'resources/a2_depth_partc_recovery_cpu_20261101.json')
        checks.record('cpu_validation',evidence['status']=='PASS' and all(digest(frozen/n)==v for n,v in evidence['files'].items()),evidence)
    checks.attempt('cpu_validation_receipt',cpu)
    cfg.update(results_root=str(root),code_commit=commit,source_manifest=lock(root.parents[1]/'source-manifest.json'))
    return write(root/'config/plan.json',cfg)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--session',required=True);p.add_argument('--code-commit',required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--recovery-handoff',type=Path)
    a=p.parse_args()
    if a.resume:
        if not a.recovery_handoff:p.error('--resume requires a fresh --recovery-handoff JSON')
        from scripts.a2_depth_continue import continue_run
        return continue_run(a)
    source=Path(__file__).resolve().parents[1];cfg=read(a.plan);started=time.monotonic()
    if socket.gethostname()!='thiennh-p6-6y3o-worker-0' or a.output.exists():raise ValueError('Wrong host or duplicate root')
    for session in (a.session,a.session+'_burn'):
        if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Duplicate session')
    admission=predecessor(cfg,a.output)
    a.output.mkdir();plan_rel=a.plan.resolve().relative_to(source)
    frozen=freeze_source(source,a.output,a.code_commit,extra_files=(str(plan_rel),'docs/A2_depth_spec.md',
        'resources/a2_depth_partc_integration_cpu_20261101.json','resources/a2_depth_partc_recovery_cpu_20261101.json','resources/a2_depth_partc_cpu_tests_20261101.json',
        'resources/a2_depth_resume_distributed_cpu_20261101.json','resources/a2_depth_stage1_references_20261101.json',
        'resources/a2_depth_stream_reference_20261101.json'))
    root=a.output/'results/a2_depth_20261031';root.mkdir(parents=True)
    config=prepare(cfg,root,frozen,a.code_commit)
    write(a.output/'admission.json',admission)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
         'HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
         'MKL_NUM_THREADS':'1','NCCL_NVLS_ENABLE':'0','TOKENIZERS_PARALLELISM':'false','PYTHONPATH':str(frozen),
         'A2_CODE_COMMIT':a.code_commit}
    from accelerate.commands.config.config_args import default_config_file,load_config_from_file
    cache=Path(default_config_file);cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():shutil.copyfile(cache,a.output/'previous-accelerate.yaml')
    expected=frozen/'resources/accelerate_b200.yaml';shutil.copyfile(expected,cache)
    actual=load_config_from_file(str(cache));checks=Checks(root/'runtime_checks.json')
    checks.record('accelerate_configuration',cache.read_bytes()==expected.read_bytes() and actual.num_processes==8 and actual.mixed_precision=='bf16' and not actual.use_cpu)
    result=subprocess.run([sys.executable,'-m','accelerate.commands.env'],capture_output=True,text=True,
                          env={**env,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
    (a.output/'accelerate-env.log').write_text(result.stdout+result.stderr)
    checks.record('accelerate_env',result.returncode==0)
    write(a.output/'budget.json',dict(preparation_seconds=time.monotonic()-started,expected_node_hours=39,
                                     timeout_seconds=None,deadline=None,policy='report-only'))
    admission=predecessor(cfg,a.output);write(a.output/'inspection.json',admission['inspection'])
    jobs=write(a.output/'jobs.json',job_config(config))
    keys=('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY',
          'WANDB_MODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NCCL_NVLS_ENABLE','TOKENIZERS_PARALLELISM','PYTHONPATH','A2_CODE_COMMIT')
    cmd=['env',*[k+'='+env[k] for k in keys],sys.executable,'-u','-m','scripts.train_then_burn','--config',jobs['path'],
         '--output',str(a.output/'supervised'),'--inspection',str(a.output/'inspection.json'),'--host',socket.gethostname(),
         '--burn-session',a.session+'_burn','--predecessor-burn-script',cfg['predecessor']['burn_script']]
    write(a.output/'launch.json',dict(session=a.session,config=config,jobs=jobs,input_plan=lock(frozen/plan_rel),
        source_manifest=lock(a.output/'source-manifest.json'),initial_inspection=lock(a.output/'inspection.json'),
        environment={k:env[k] for k in keys}))
    command='exec '+shlex.join(cmd)+' >'+shlex.quote(str(a.output/'supervisor.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',a.session,'-c',str(frozen),command],env=env,check=True)
    subprocess.run(['tmux','set-option','-w','-t',a.session,'remain-on-exit','on'],check=True)
    print('A2_DEPTH_PART_C_SUBMITTED',a.output,flush=True)


if __name__=='__main__':main()
