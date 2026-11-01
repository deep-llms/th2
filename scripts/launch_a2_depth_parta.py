"""Launch the authorized A2 Depth Part A using the unchanged native GPU supervisor."""
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

from scripts.a2_quant_io import lock, write
from scripts.a2_remaining import read_locked
from scripts.launch_a2_downstream import freeze_source
from scripts.gpu_process import inspect, process
from scripts.train_then_burn import digest, argv, BURN_HASH, GUARD_HASH


def predecessor(cfg, output):
    """Physical ownership remains mandatory before any signal or allocation."""
    b=cfg['predecessor'];root=Path(b['root']);script=Path(b['burn_script'])
    if root.parent.resolve()!=output.parent.resolve() or script!=root/'source/resources/llm_pretrain_burn.py':
        raise ValueError('Wrong authorized physical predecessor')
    supervisor=read_locked(b['supervisor']);burn=read_locked(b['burn']);done=read_locked(b['complete'])
    if supervisor['training_status']!=b['prior_status'] or not supervisor.get('finished_at') or not done['execution_complete'] or not done['coverage_complete'] or not done['readout_complete']:
        raise ValueError('Prior registered workload is not terminal')
    if not burn['collective_progress_verified']:raise ValueError('Prior native handoff not verified')
    source=read_locked(b['source_manifest'])
    for rel,value in source['files'].items():
        if digest(root/'source'/rel)!=value:raise ValueError('Physical predecessor source changed')
    current=inspect()
    expected={(g['index'],g['uuid'],tuple(g['pids'])) for g in burn['gpus']}
    actual={(g['index'],g['uuid'],tuple(g['pids'])) for g in current['gpus']}
    if expected!=actual:raise ValueError('Different live GPU workers; no signal sent')
    records={}
    for g in current['gpus']:
        for pid in g['pids']:
            worker=process(pid);parent=process(worker['ppid'])
            if str(script) not in argv(parent['pid']):raise ValueError('Different burn command')
            records[str(pid)]=worker;records[str(parent['pid'])]=parent
    if records!=b['known_burn_identities']:raise ValueError('Burn start/command identity changed')
    if digest(script)!=BURN_HASH or digest('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH:
        raise ValueError('Unknown supervisor/guard ownership')
    if Path('/mnt/local/_gpu_guard/DISABLED').exists():raise ValueError('Another native guard owner')
    return dict(inspection=current,identities=records,predecessor=b)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True);p.add_argument('--session',required=True)
    p.add_argument('--code-commit',required=True);a=p.parse_args()
    started=time.monotonic();source=Path(__file__).resolve().parents[1]
    cfg=json.loads((source/'resources/a2_depth_parta_plan_20261101.json').read_text())
    if socket.gethostname()!='thiennh-p6-6y3o-worker-0' or a.output.exists():
        raise ValueError('Wrong host or duplicate output')
    for session in (a.session,a.session+'_burn'):
        if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:
            raise ValueError('Duplicate session')
    admission=predecessor(cfg,a.output)
    a.output.mkdir()
    frozen=freeze_source(source,a.output,a.code_commit,extra_files=(
        'docs/A2_depth_spec.md','resources/a2_depth_parta_plan_20261101.json',
        'resources/a2_depth_parta_cpu_tests_20261101.json'))
    root=a.output/'results/a2_depth_20261031';root.mkdir(parents=True)
    (root/'manifests').mkdir()
    checks=[]
    def record(name,passed,detail):
        event=dict(test=name,passed=bool(passed),detail=detail,spec_sha256=cfg['spec_sha256'])
        checks.append(event);write(root/'preparation_checks.json',dict(spec_sha256=cfg['spec_sha256'],checks=checks))
        print('DEPTH_PART_A_PREPARATION',json.dumps(event),flush=True)
    from efficient_llm.a2.data import PackedRows
    def attempt(name,fn):
        try:
            value=fn();record(name,True,'Verified');return value
        except Exception as error:
            record(name,False,dict(error=repr(error),action='continue registered independent probes'));return None
    record('spec_identity',digest(source/'docs/A2_depth_spec.md')==cfg['spec_sha256'],cfg['spec_sha256'])
    manifest=attempt('data_manifest',lambda:read_locked(cfg['data_manifest']))
    record('small_set_identity',manifest is not None and manifest['small_indices']==cfg['small_indices'],dict(count=len(cfg['small_indices'])))
    for stream in ('tuning','final'):
        attempt('packed_data:'+stream,lambda stream=stream:PackedRows(cfg['data_root'],stream,verify=True))
    attempt('assets',lambda:read_locked(cfg['assets']))
    for name,item in cfg['reference_nll'].items():
        attempt('reference_nll:'+name,lambda item=item:read_locked(item))
    for ds,item in cfg['heal_streams'].items():
        attempt('heal_manifest:'+ds,lambda item=item:read_locked(dict(path=item['path']+'/manifest.json',sha256=item['sha256'])))
    recorded_sets=attempt('phase0_sets',lambda:read_locked(cfg['phase0_sets']))
    cfg.update(code_commit=a.code_commit,results_root=str(root),source_manifest=lock(a.output/'source-manifest.json'))
    # Freeze evidence beside the config, not paths into the Git-synchronized checkout.
    shutil.copyfile(frozen/'docs/A2_depth_spec.md',root/'config-spec.md')
    shutil.copyfile(frozen/'resources/a2_depth_parta_cpu_tests_20261101.json',root/'manifests/cpu_tests.json')
    cfg_lock=write(root/'config/plan.json',cfg)
    write(root/'manifests/phase0_sets.json',dict(spec_sha256=cfg['spec_sha256'],recorded=recorded_sets,
          lock=cfg['phase0_sets'],fallback_layer_sets=cfg['layer_sets'] if recorded_sets is None else None))
    write(root/'manifests/input_locks.json',dict(spec_sha256=cfg['spec_sha256'],assets=cfg['assets'],data_manifest=cfg['data_manifest'],heal_streams=cfg['heal_streams'],phase0_sets=cfg['phase0_sets']))
    write(a.output/'admission.json',admission)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline',
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NCCL_NVLS_ENABLE':'0',
        'TOKENIZERS_PARALLELISM':'false','PYTHONPATH':str(frozen),'A2_CODE_COMMIT':a.code_commit}
    write(a.output/'budget.json',dict(spec_sha256=cfg['spec_sha256'],preparation_seconds=time.monotonic()-started,
          expected_node_hours=.6,timeout_seconds=None,deadline=None,policy='report-only'))
    # Fresh physical identities again after CPU preparation; native supervisor rechecks via pidfds.
    admission=predecessor(cfg,a.output)
    write(a.output/'inspection.json',admission['inspection'])
    jobs=write(a.output/'jobs.json',dict(jobs=[dict(name='a2-depth-parta',gpus=list(range(8)),timeout_seconds=None,
        argv=['{python}','-u','-m','scripts.a2_depth_parta_run','--config',cfg_lock['path'],
              '--receipt','{run_dir}/depth-parta-complete.json'],
        required_outputs=[dict(path='depth-parta-complete.json',json_equals=dict(execution_complete=True))])]))
    cmd=['env',*[k+'='+env[k] for k in ('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
        'HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','WANDB_MODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS','NCCL_NVLS_ENABLE','TOKENIZERS_PARALLELISM','PYTHONPATH','A2_CODE_COMMIT')],
        sys.executable,'-u','-m','scripts.train_then_burn','--config',jobs['path'],
        '--output',str(a.output/'supervised'),'--inspection',str(a.output/'inspection.json'),
        '--host',socket.gethostname(),'--burn-session',a.session+'_burn',
        '--predecessor-burn-script',cfg['predecessor']['burn_script']]
    command='exec '+shlex.join(cmd)+' >'+shlex.quote(str(a.output/'supervisor.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',a.session,'-c',str(frozen),command],env=env,check=True)
    subprocess.run(['tmux','set-option','-w','-t',a.session,'remain-on-exit','on'],check=True)
    print('A2_DEPTH_PART_A_SUBMITTED',a.output,flush=True)


if __name__=='__main__':main()
