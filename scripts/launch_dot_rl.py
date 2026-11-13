"""Admit DOT-RL on the explicitly authorized wc69 node, then reuse owned supervisor."""
import argparse,json,os,shlex,shutil,socket,subprocess,sys,time
from pathlib import Path
from importlib.metadata import version
from efficient_llm.dot_pool.rl import SPEC_SHA,sha,write,make_data
from scripts.gpu_process import inspect,process
from scripts.dot_pool_report_recovery import check_burn
from scripts.train_then_burn import BURN_HASH,GUARD_HASH
from run_experiments import load_jobs

HOST='thiennh-p6-wc69-worker-0'
BASE=Path('/mnt/local/_outputs/deep-llms_th2/dot_rl_20261114')
BURN=Path('/mnt/local/_outputs/deep-llms_th2/dot_pool_20261112/execution-e1b-v15-a02/source/resources/llm_pretrain_burn.py')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--commit',required=True);a=ap.parse_args()
    source=Path(__file__).resolve().parents[1];execution=BASE/'execution-a01';run=BASE/'results/dot_rl_20261114/run-a01';session='dot_rl_20261114_a01'
    if socket.gethostname()!=HOST or len(a.commit)!=40 or any(c not in '0123456789abcdef' for c in a.commit):raise ValueError('Node/commit mismatch')
    if execution.exists() or run.exists():raise ValueError('Existing execution/run; retrieve receipt, never relaunch')
    if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Session exists')
    if sha(source/'docs/DOT_RL_router_spec.md')!=SPEC_SHA:raise ValueError('Spec changed')
    runtime={}
    for line in (source/'envs/dot_pool.txt').read_text().splitlines():
        if '==' in line:
            package,expected=line.split('==');package=package.split('[')[0];actual=version(package);runtime[package]=actual
            if actual.split('+')[0]!=expected:raise ValueError('Dependency mismatch '+package)
    pin=json.loads((source/'resources/dot_rl_admission.json').read_text())
    for rel,digest in pin['files'].items():
        if sha(source/rel)!=digest:raise ValueError('Source changed '+rel)
    ownership=check_burn(pin['burn'])
    if sha(BURN)!=BURN_HASH or sha('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH or Path('/mnt/local/_gpu_guard/DISABLED').exists():raise ValueError('Burn/guard mismatch')
    # Generate the NEW split offline, independent of completed E1b data/outputs.
    data=run.parent/'data'
    if not data.exists():make_data(data)
    expected=json.loads((source/'resources/dot_rl_data_expected.json').read_text())
    if json.loads((data/'manifest.json').read_text())!=expected:raise ValueError('Local/node split manifest mismatch')
    for name,r in expected['files'].items():
        if sha(data/name)!=r['sha256']:raise ValueError('Actual data SHA mismatch '+name)
    execution.mkdir(parents=True);frozen=execution/'source'
    files=list(source.glob('*.py'))
    for folder in ['efficient_llm','scripts','eval','tests']:files+=list((source/folder).rglob('*.py'))
    files += [source/p for p in ['resources/llm_pretrain_burn.py','resources/dot_rl_admission.json','resources/dot_rl_data_expected.json','docs/DOT_RL_router_spec.md','envs/dot_pool.txt']]
    hashes={}
    for path in sorted(set(files)):
        rel=path.relative_to(source);target=frozen/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);hashes[str(rel)]=sha(target)
    write(execution/'source-manifest.json',dict(commit=a.commit,files=hashes))
    write(execution/'admission.json',dict(runtime=runtime,data_manifest_sha256=sha(data/'manifest.json'),data_content_verified=True,ownership=ownership,utc=time.time()))
    write(execution/'inspection.json',check_burn(pin['burn']))
    jobs=dict(jobs=[dict(name='dot-rl-v12',gpus=list(range(8)),argv=['{python}','-u','-m','scripts.dot_rl_queue','--root',str(run),'--receipt','{run_dir}/dot-rl-result.json'],required_outputs=[dict(path='dot-rl-result.json',json_equals=dict(reported=True))])])
    (execution/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n');load_jobs(execution/'jobs.json')
    env={**os.environ,'PYTHONPATH':str(frozen),'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline','OMP_NUM_THREADS':'2','NCCL_NVLS_ENABLE':'0'}
    argv=[sys.executable,'-u','-m','scripts.train_then_burn','--config',str(execution/'jobs.json'),'--output',str(execution/'supervised'),'--inspection',str(execution/'inspection.json'),'--host',HOST,'--burn-session',session+'_burn','--predecessor-burn-script',str(BURN)]
    keys=['PYTHONPATH','CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','WANDB_MODE','OMP_NUM_THREADS','NCCL_NVLS_ENABLE']
    command='exec '+shlex.join(['env',*[k+'='+env[k] for k in keys],*argv])+' >'+shlex.quote(str(execution/'supervisor.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),command],env=env,check=True)
    subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
    write(execution/'submitted.json',dict(session=session,run=str(run),commit=a.commit,utc=time.time()))
    print('DOT_RL_SUBMITTED',json.dumps(dict(session=session,run=str(run),source_manifest_sha256=sha(execution/'source-manifest.json'),data_sha256=sha(data/'manifest.json'),runtime=runtime,verified_prior_workers=ownership['workers'])),flush=True)

if __name__=='__main__':main()
