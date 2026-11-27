"""Freeze registered sources and start the supervised long study on the allocated node."""
import json
from pathlib import Path
import shutil
import shlex
import socket
import subprocess
import sys

from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.horizon_train import write_json

ROOT=Path('/mnt/local/_outputs/deep-llms_th2')
OLD=ROOT/'embedding_anchor_fit_20261122_a01/execution'
PRIOR='9d858eff601bbd3a30a07d53297753f277a074885a3dae113733ebec0e11eb3b'
BASE=ROOT/'anchor_long_20261122_a01'
ADDED=('scripts/anchor_long_queue.py','scripts/anchor_long_study.py',
       'scripts/anchor_boundary_study.py','efficient_llm/warm_order/anchor_long.py')


def launch(extra, loop=False, static=False):
    base = ROOT/'anchor_loop_20261123_a01' if loop else BASE
    old = BASE/'execution' if loop else OLD
    prior_sha = 'c8bbce6e52c5c6a6b6a23506e217c0173570804fcb5c81fe720ea614c8470cf8' if loop else PRIOR
    changes = ('scripts/anchor_long_queue.py','scripts/anchor_loop_study.py',
               'efficient_llm/warm_order/anchor_loop.py') if loop else ADDED
    replaced = {'scripts/anchor_long_queue.py'} if loop else set()
    if static:
        base = ROOT/'anchor_static_20261123_a01'
        old = ROOT/'anchor_loop_20261123_a01/execution'
        prior_sha = '504ff4fdee7bf160e4b96bd04a782e50fd57ba7a07f13857731ff87bf62afb24'
        changes = ('scripts/anchor_loop_study.py', 'efficient_llm/warm_order/anchor_loop.py')
        replaced = set(changes)
    if socket.gethostname()!='thiennh-p6-ohmv-worker-0' or sha(old/'source.json')!=prior_sha:
        raise ValueError('Unexpected node or ancestor')
    prior=json.loads((old/'source.json').read_text())
    pins=json.loads((extra/'pins.json').read_text())
    if set(pins)!=set(changes) or (set(pins)&set(prior))!=replaced:
        raise ValueError('Invalid registered delta')
    if shutil.disk_usage(ROOT).free < 100 * 1024**3:
        raise ValueError('Insufficient disk space')
    base.mkdir(exist_ok=False)
    execution=base/'execution'
    frozen=execution/'source'
    for name,expected in (prior|pins).items():
        src=(extra if name in pins else old/'source')/name
        if sha(src)!=expected:
            raise ValueError('Source changed: '+name)
        dst=frozen/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
        if sha(dst)!=expected:
            raise ValueError('Frozen copy mismatch: '+name)
    write_json(execution/'source.json',prior|pins)
    env=['PYTHONPATH='+str(frozen),'PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1',
         'TRANSFORMERS_OFFLINE=1','CUBLAS_WORKSPACE_CONFIG=:4096:8','OMP_NUM_THREADS=4',
         'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7']
    worker=['scripts.anchor_loop_study','queue'] if loop else ['scripts.anchor_long_queue']
    if static:
        worker = ['scripts.anchor_loop_study','static-queue']
    command='exec '+shlex.join(['env',*env,sys.executable,'-u','-m',*worker,
                                '--base',str(base)])+' >'+shlex.quote(str(execution/'queue.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',base.name,'-c',str(frozen),command],check=True)
    print('ANCHOR_LONG_QUEUE_SUBMITTED',sha(execution/'source.json'),flush=True)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('extra', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--loop', action='store_true')
    modes.add_argument('--static', action='store_true')
    args = parser.parse_args()
    launch(args.extra, loop=args.loop, static=args.static)
