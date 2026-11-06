"""Family -> verified scientific backup -> Depth under one native GPU owner."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
from efficient_llm.a2_family.common import read,write,sha,SPEC_SHA
from scripts.a2_remaining import read_locked
from scripts.a2_quant_io import lock,write as native_write
from scripts.launch_a2_downstream import freeze_source
from scripts.gpu_process import inspect,process
from scripts.train_then_burn import argv,BURN_HASH,GUARD_HASH
from run_experiments import load_jobs


def jobs(config):
    return dict(jobs=[dict(name='a2-family',gpus=list(range(8)),timeout_seconds=None,
        argv=['{python}','-u','-m','scripts.a2_family_run','--config',str(config),'--receipt','{run_dir}/family-complete.json'],
        required_outputs=[dict(path='family-complete.json',json_equals=dict(execution_complete=True,readout_complete=True))]),
        dict(name='a2-depth-resume-after-family',gpus=list(range(8)),timeout_seconds=None,
        argv=['{python}','-u','-m','scripts.a2_family_depth_bridge','--config',str(config),'--receipt','{run_dir}/depth-complete.json'],
        required_outputs=[dict(path='depth-complete.json',json_equals=dict(execution_complete=True))])])


def predecessor(cfg):
    b=cfg['predecessor'];root=Path(b['root']);depth=Path(cfg['depth_root'])
    if root.parent!=depth/'continuations' or Path(b['burn_script'])!=root/'source/resources/llm_pretrain_burn.py':raise ValueError('Unregistered Depth predecessor')
    terminal=read_locked(b['supervisor']);burn=read_locked(b['burn']);pause=read_locked(cfg['depth_pause'])
    if pause['state']!='paused' or not pause['handoff_verified'] or not terminal.get('finished_at') or terminal['training_status']=='running' or not burn['collective_progress_verified']:raise ValueError('Depth pause/handoff not verified')
    if Path('/mnt/local/_gpu_guard/DISABLED').exists():raise ValueError('Native guard still owned')
    if sha(b['burn_script'])!=BURN_HASH or sha('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH:raise ValueError('Burn/guard code changed')
    for rel,h in read_locked(b['source_manifest'])['files'].items():
        if sha(root/'source'/rel)!=h:raise ValueError('Predecessor frozen source changed')
    current=inspect();expected={(g['index'],g['uuid'],tuple(g['pids'])) for g in burn['gpus']}
    if expected!={(g['index'],g['uuid'],tuple(g['pids'])) for g in current['gpus']}:raise ValueError('GPU ownership changed')
    identities={}
    for g in current['gpus']:
        for pid in g['pids']:
            worker=process(pid);parent=process(worker['ppid'])
            if b['burn_script'] not in argv(parent['pid']):raise ValueError('Unknown burn launcher')
            identities[str(pid)]=worker;identities[str(parent['pid'])]=parent
    if identities!=b['known_burn_identities']:raise ValueError('Burn start/command changed')
    return current


def verify_inputs(cfg,paused=True):
    from efficient_llm.a2.data import PackedRows
    if read_locked(cfg['models_manifest'])!=cfg['models']:raise ValueError('Model manifest binding differs')
    cx_ready=read_locked(cfg['cx_complete'])
    read_locked(cfg['opensci_manifest'])
    for entry in cfg['models']['checkpoints']:
        for f in entry['files']:
            p=Path(entry['path'])/f['name']
            if p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:raise ValueError('Model identity differs: '+str(p))
    for split in ('tuning','final'):
        root=Path(cfg['cx_root'])/split;m=read(root/'manifest.json')
        if any(m.get(k)!=v for k,v in cx_ready['sets'][split].items()):raise ValueError('CX manifest binding differs')
        for name,key in [('tokens.npy','token_sha256'),('offsets.npy','offset_sha256'),('documents.json','documents_sha256')]:
            if sha(root/name)!=m[key]:raise ValueError('CX identity differs')
        PackedRows(cfg['opensci_cx_root'],split,verify=True)
    for item in [*cfg['streams'].values(),cfg['mixture_stream']]:
        read_locked(dict(path=str(Path(item['path'])/'manifest.json'),sha256=item['sha256']))
        PackedRows(item['path'],'heal',verify=True)
    if paused:
        from scripts.a2_family_depth_bridge import validate
        validate(cfg)


def main():
    p=argparse.ArgumentParser()
    for k in ('plan','output','session','code-commit'):p.add_argument('--'+k,required=True)
    a=p.parse_args();cfg=read(a.plan);out=Path(a.output);source=Path(__file__).resolve().parents[1]
    if socket.gethostname()!='thiennh-p6-wc69-worker-0' or out.exists() or cfg['spec_sha256']!=SPEC_SHA:raise ValueError('Wrong node/spec or existing output')
    for session in (a.session,a.session+'_burn'):
        if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Existing session')
    proof_path='resources/a2_family_integration_cpu_20261107.json';proof=read(source/proof_path)
    if proof['status']!='PASS' or not proof.get('launch_ready'):raise ValueError('Full pipeline CPU admission not ready')
    for rel,h in proof['files'].items():
        if sha(source/rel)!=h:raise ValueError('CPU proof changed')
    predecessor(cfg);verify_inputs(cfg);out.mkdir(parents=True,exist_ok=False)
    extra=['docs/A2_family_spec_v1.3.md',proof_path,*[f for f in proof['files'] if f.startswith(('tests/','resources/'))]]
    frozen=freeze_source(source,out,a.code_commit,extra_files=extra)
    if sha(frozen/'docs/A2_family_spec_v1.3.md')!=SPEC_SHA:raise ValueError('Spec changed')
    root=out/'results/a2_family_20261107';root.mkdir(parents=True)
    cfg.update(results_root=str(root),code_commit=a.code_commit,source_manifest=lock(out/'source-manifest.json'))
    from scripts.a2_family_overlap import audit
    cfg['overlap']=audit(cfg,root/'overlap_audit/pools.json')
    config=write(root/'config/plan.json',cfg);shutil.copyfile(frozen/'docs/A2_family_spec_v1.3.md',root/'config/spec.md')
    write(root/'preparation_checks.json',dict(checks=[dict(test='actual_input_hashes',passed=True),dict(test='FT7_LR_state',passed=cfg['peak_lr']==2e-4,detail=cfg['lr_state'])]))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NCCL_NVLS_ENABLE':'0','TOKENIZERS_PARALLELISM':'false','PYTHONPATH':str(frozen),'A2_CODE_COMMIT':a.code_commit}
    from accelerate.commands.config.config_args import default_config_file,load_config_from_file
    cache=Path(default_config_file);cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():shutil.copyfile(cache,out/'previous-accelerate.yaml')
    expected=frozen/'resources/accelerate_b200.yaml';shutil.copyfile(expected,cache);actual=load_config_from_file(str(cache))
    if cache.read_bytes()!=expected.read_bytes() or actual.num_processes!=8 or actual.mixed_precision!='bf16' or actual.use_cpu:raise ValueError('Accelerate config mismatch')
    test=subprocess.run([sys.executable,'-m','accelerate.commands.env'],capture_output=True,text=True,env=env);(out/'accelerate-env.log').write_text(test.stdout+test.stderr)
    if test.returncode:raise ValueError('Accelerate runtime unavailable')
    inspection=write(out/'inspection.json',predecessor(cfg));job=native_write(out/'jobs.json',jobs(config['path']));load_jobs(Path(job['path']))
    cmd=[sys.executable,'-u','-m','scripts.train_then_burn','--config',job['path'],'--output',str(out/'supervised'),'--inspection',inspection['path'],'--host',socket.gethostname(),'--burn-session',a.session+'_burn','--predecessor-burn-script',cfg['predecessor']['burn_script']]
    command='exec '+shlex.join(cmd)+' >'+shlex.quote(str(out/'supervisor.log'))+' 2>&1'
    write(out/'launch.json',dict(config=config,jobs=job,session=a.session,automatic_Depth_resume=True,source_manifest=cfg['source_manifest']))
    subprocess.run(['tmux','new-session','-d','-s',a.session,'-c',str(frozen),command],env=env,check=True)
    subprocess.run(['tmux','set-option','-w','-t',a.session,'remain-on-exit','on'],check=True)
    print('FAMILY_NATIVE_QUEUE_SUBMITTED',out,flush=True)

if __name__=='__main__':main()
