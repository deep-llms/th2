"""CPU-only replay of immutable FP32 scores; never launch model/GPU work."""
import argparse, concurrent.futures, json, os, shlex, shutil, subprocess, sys, time
from pathlib import Path
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.train import write_json
from efficient_llm.warm_order.downstream_report import compact, report, TASKS
from scripts.gpu_process import inspect, ownership

OLD=Path('/mnt/local/_outputs/deep-llms_th2/warm_downstream_20261118_a02')
BASE=Path('/mnt/local/_outputs/deep-llms_th2/warm_downstream_replay_20261118_a01')
PIN='resources/warm_downstream_recovery_source_20261118.json'
EVAL='/mnt/local/conda-py311/envs/eval/bin/python3.11'

def verify_original():
    terminal=json.loads((OLD/'terminal.json').read_text())
    if terminal['success'] or not terminal['native_verified']:raise ValueError('Require failed/released original queue')
    for rel,w in json.loads((OLD/'execution/source.json').read_text()).items():
        if sha(OLD/'execution/source'/rel)!=w:raise ValueError('Original source changed')
    plan=json.loads((OLD/'plan.json').read_text())
    if sha(plan['requests'])!=plan['requests_sha256']:raise ValueError('Request bytes changed')
    if not json.loads((OLD/'gpu-probe.json').read_text())['passed']:raise ValueError('GPU probe not passed')
    evidence={}
    for e in plan['models']:
        for p,w in e['hashes'].items():
            if sha(p)!=w:raise ValueError('Checkpoint changed')
        run=OLD/'runs'/e['name'];done=json.loads((run/'complete.json').read_text())
        if done['requests_sha256']!=plan['requests_sha256']:raise ValueError('Wrong score inputs')
        for p,w in done['files'].items():
            if sha(run/p)!=w:raise ValueError('Score member changed')
        evidence[e['name']]={p:sha(run/p) for p in ('scores.json','details.json','complete.json')}
    return plan,evidence

def launch(commit):
    if BASE.exists():raise ValueError('Existing recovery; retrieve, do not relaunch')
    pin=json.loads(Path(PIN).read_text())['files']
    for rel,w in pin.items():
        if sha(rel)!=w:raise ValueError('Recovery source changed')
    plan,evidence=verify_original();current=inspect()
    if current['guard_disabled'] or len(current['workers'])!=8:raise ValueError('Native handoff not intact')
    BASE.mkdir();frozen=BASE/'recovery-source'
    for rel,w in pin.items():
        dst=frozen/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(rel,dst)
        if sha(dst)!=w:raise ValueError('Recovery freeze mismatch')
    (BASE/'execution').mkdir()
    for rel in ('execution/source.json','execution/admission.json','gpu-probe.json'):
        shutil.copyfile(OLD/rel,BASE/rel)
    write_json(BASE/'recovery-source.json',pin)
    write_json(BASE/'plan.json',plan)
    for name,files in evidence.items():
        target=BASE/'runs'/name;target.mkdir(parents=True)
        for rel,w in files.items():
            shutil.copyfile(OLD/'runs'/name/rel,target/rel)
            if sha(target/rel)!=w:raise ValueError('Score copy mismatch')
    write_json(BASE/'recovery.json',dict(commit=commit,original=str(OLD),original_terminal_sha256=sha(OLD/'terminal.json'),score_hashes=evidence,ownership=current,utc=time.time(),score_precision='fp32_no_autocast_no_tf32',no_gpu_work=True))
    env=['CUDA_VISIBLE_DEVICES=','HF_HUB_OFFLINE=1','HF_DATASETS_OFFLINE=1','TRANSFORMERS_OFFLINE=1','HF_HUB_DISABLE_TELEMETRY=1','WANDB_MODE=offline','TOKENIZERS_PARALLELISM=false','OMP_NUM_THREADS=4','PYTHONPATH='+str(frozen)]
    cmd='exec '+shlex.join(['env',*env,sys.executable,'-u','-m','scripts.warm_downstream_recover','run'])+' >'+shlex.quote(str(BASE/'recovery.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s','warm_downstream_replay_20261118_a01','-c',str(frozen),cmd],check=True)
    print('DOWNSTREAM_RECOVERY_SUBMITTED',commit,flush=True)

def replay(e):
    run=BASE/'runs'/e['name']
    argv=[EVAL,'-m','eval.a2_requests','replay','--tokenizer','/mnt/local/_data/deep-llms_th2/a2_dot_20261118/tokenizer','--dataset-root','/mnt/local/_data/deep-llms_th2/warm_downstream_20261118','--manifest','resources/warm_downstream_data_20261118.json','--tasks',*TASKS,'--max-length','512','--prepared',str(OLD/'requests.json'),'--scored',str(run/'scores.json'),'--score-precision','fp32_no_autocast_no_tf32','--output',str(run/'raw.json')]
    with (run/'replay.log').open('x') as log:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
    items=compact(json.loads((run/'raw.json').read_text()),json.loads((OLD/'requests.json').read_text()),json.loads((run/'scores.json').read_text()))
    write_json(run/'items.json',items)
    write_json(run/'replay-complete.json',dict(items_sha256=sha(run/'items.json'),raw_sha256=sha(run/'raw.json'),tasks={t:dict(accuracy=v['accuracy'],items=len(v['items'])) for t,v in items.items()}))
    print('REPLAY_OK',e['name'],flush=True)

def run():
    try:
        verify_original()
        pin=json.loads((BASE/'recovery-source.json').read_text())
        for rel,w in pin.items():
            if sha(rel)!=w:raise ValueError('Recovery source changed')
        plan=json.loads((BASE/'plan.json').read_text())
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(replay,plan['models']))
        proof=report(BASE)
        now=inspect();before=json.loads((BASE/'recovery.json').read_text())['ownership']
        if ownership(now['gpus'])!=ownership(before['gpus']) or now['guard_disabled']:raise ValueError('Native ownership changed during CPU recovery')
        write_json(BASE/'terminal.json',dict(success=True,utc=time.time(),ownership=now,archive_sha256=proof['archive_sha256'],no_gpu_work=True))
    except BaseException as e:
        write_json(BASE/'failure.json',dict(error=repr(e),utc=time.time()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['launch','run']);p.add_argument('--commit');a=p.parse_args()
    launch(a.commit) if a.mode=='launch' else run()
