"""One native v1.7 queue, after independently pinned Depth pause and known burn."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
from efficient_llm.a2_family_v17.common import read,write,SPEC_SHA
from efficient_llm.a2_family_v17.protocol import runs
from scripts.a2_family_v14_gate import sha,read_lock
from scripts.a2_quant_io import lock,write as native_write
from scripts.launch_a2_downstream import freeze_source
from scripts.gpu_process import inspect,process
from scripts.train_then_burn import argv,BURN_HASH,GUARD_HASH
from scripts.a2_family_v17_prepare import verify_inputs
from run_experiments import load_jobs


def jobs(config):
    return dict(jobs=[dict(name='family-v17-F8-F7',gpus=list(range(8)),timeout_seconds=None,
        argv=['{python}','-u','-m','scripts.a2_family_v17_run','--config',str(config),'--receipt','{run_dir}/v17-complete.json'],
        required_outputs=[dict(path='v17-complete.json',json_equals=dict(execution_complete=True,report_complete=True,backup_verified=True))]),
        dict(name='depth-after-v17',gpus=list(range(8)),timeout_seconds=None,
        argv=['{python}','-u','-m','scripts.a2_family_v17_depth_bridge','--config',str(config),'--receipt','{run_dir}/depth-complete.json'],
        required_outputs=[dict(path='depth-complete.json',json_equals=dict(execution_complete=True))])])


def predecessor(cfg):
    root=Path(cfg['predecessor_root'])
    if str(root)!='/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-20261107-a01':raise ValueError('Wrong predecessor')
    pause=read_lock(cfg['depth_pause']);terminal=read(root/'supervised/supervisor.json');burn=read(root/'supervised/burn-verified.json')
    if pause['state']!='paused' or not pause['handoff_verified'] or pause['supervisor']!=terminal or not terminal.get('finished_at') or not burn['collective_progress_verified']:
        raise ValueError('Pause/native handoff incomplete')
    if Path('/mnt/local/_gpu_guard/DISABLED').exists():raise ValueError('Guard still owned')
    script=root/'source/resources/llm_pretrain_burn.py'
    if sha(script)!=BURN_HASH or sha('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH:raise ValueError('Changed burn/guard')
    live=inspect()
    if live['host']!=cfg['host']:raise ValueError('Wrong physical node')
    mapping=lambda gs:{(g['index'],g['uuid'],tuple(g['pids'])) for g in gs}
    if mapping(live['gpus'])!=mapping(burn['gpus']):raise ValueError('Burn GPU mapping changed')
    for g in live['gpus']:
        for pid in g['pids']:
            if str(script) not in argv(process(pid)['ppid']):raise ValueError('Unknown GPU process')
    return live,str(script)


def launch(plan,output,session,commit):
    source=Path(__file__).resolve().parents[1];out=Path(output);cfg=read(plan)
    if socket.gethostname()!=cfg['host'] or cfg['spec_sha256']!=SPEC_SHA or out.exists():raise ValueError('Wrong node/spec or existing root')
    for name in (session,session+'_burn'):
        if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0:raise ValueError('Existing session')
    proof=read(source/'resources/a2_family_v17_cpu_20261109.json')
    if proof['status']!='PASS' or not proof['launch_ready'] or proof['spec_sha256']!=SPEC_SHA:raise ValueError('CPU admission incomplete')
    for path,h in proof['files'].items():
        if sha(source/path)!=h:raise ValueError('Changed tested source: '+path)
    pause=Path(cfg['pause_root']);print('V17_WAIT_PAUSE',pause,flush=True)
    while not (pause/'terminal.json').exists():
        # The pause helper is deliberately not restarted here.
        proc=subprocess.run(['tmux','display-message','-p','-t','a2_family_v17_pause_20261109_a01','#{pane_dead}'],capture_output=True,text=True)
        if proc.returncode or proc.stdout.strip()!='0':raise ValueError('Pause helper exited without terminal; inspect original log')
        time.sleep(10)
    cfg['depth_pause']=lock(pause/'terminal.json')
    depth=Path(cfg['depth_root']);original=read(depth/'launch.json');current=read(depth/'continuations.json')['attempts'][-1]
    cfg.update(depth_launch=lock(depth/'launch.json'),depth_execution_source=str(Path(current['root'])/'source'),depth_execution_manifest=current['execution_source_manifest'])
    predecessor(cfg)
    out.mkdir(parents=True,exist_ok=False)
    root=out/'results/a2_family_20261107';root.mkdir(parents=True)
    write(out/'progress.json',dict(state='verifying_actual_inputs'))
    verified=verify_inputs(cfg)
    extras=['docs/A2_family_spec_v1.7.md','resources/a2_family_v17_plan_20261109.json','resources/a2_family_v17_cpu_20261109.json',*proof['files']]
    frozen=freeze_source(source,out,commit,extra_files=extras)
    cfg.update(results_root=str(root),source_manifest=lock(out/'source-manifest.json'),code_commit=commit,
        native_output=str(out/'supervised'),runs={r['run_id']:r for r in runs()})
    config=write(root/'config/plan.json',cfg)
    shutil.copyfile(frozen/'docs/A2_family_spec_v1.7.md',root/'config/spec.md')
    write(root/'manifests/actual_inputs.json',dict(files=verified,FT10=cfg['FT10']))
    write(root/'manifests/models.json',dict(models={a:read_lock(cfg['assets'])['dclm/'+a] for a in ('P300','P1T','S72','S188')}))
    write(root/'manifests/pause.json',read_lock(cfg['depth_pause']))
    env={**os.environ,**original['environment'],'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(frozen),'A2_CODE_COMMIT':commit}
    from accelerate.commands.config.config_args import default_config_file,load_config_from_file
    cache=Path(default_config_file);cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():shutil.copyfile(cache,out/'previous-accelerate.yaml')
    expected=frozen/'resources/accelerate_b200.yaml';shutil.copyfile(expected,cache);actual=load_config_from_file(str(cache))
    if actual.num_processes!=8 or actual.mixed_precision!='bf16' or actual.use_cpu:raise ValueError('Wrong accelerate configuration')
    inspection,script=predecessor(cfg);inspection_lock=write(out/'inspection.json',inspection)
    job=native_write(out/'jobs.json',jobs(config['path']));load_jobs(Path(job['path']))
    cmd=[sys.executable,'-u','-m','scripts.train_then_burn','--config',job['path'],'--output',str(out/'supervised'),
        '--inspection',inspection_lock['path'],'--host',cfg['host'],'--burn-session',session+'_burn','--predecessor-burn-script',script]
    command='exec '+shlex.join(cmd)+' >'+shlex.quote(str(out/'supervisor.log'))+' 2>&1'
    write(out/'launch.json',dict(config=config,jobs=job,source_manifest=cfg['source_manifest'],session=session,automatic_Depth_resume=True))
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),command],env=env,check=True)
    subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
    print('V17_QUEUE_SUBMITTED',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('plan','output','session','code-commit'):p.add_argument('--'+k,required=True)
    a=p.parse_args();launch(a.plan,a.output,a.session,a.code_commit)
