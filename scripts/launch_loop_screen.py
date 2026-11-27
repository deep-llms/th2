"""Freeze a loop-screen-family source on top of the verified loop-study source and start its supervisor.

Same pattern as scripts/launch_anchor_long.py: every prior frozen file is copied
with its recorded hash, new files must match pins.json, the output root must not
exist, and the supervisor runs in its own tmux session. GPU ownership checks and
the native-burn handoff are done by scripts.anchor_long_queue.supervise.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys

from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.horizon_train import write_json

ROOT = Path('/mnt/local/_outputs/deep-llms_th2')
OLD = ROOT/'anchor_loop_20261123_a01/execution'
PRIOR = '504ff4fdee7bf160e4b96bd04a782e50fd57ba7a07f13857731ff87bf62afb24'
ADDED = ('efficient_llm/warm_order/loop_screen.py', 'scripts/loop_screen_study.py')


def launch(extra, base, handoff, handoff_sha, old=OLD, prior_sha=PRIOR, added=ADDED,
           module='scripts.loop_screen_study', replaced=()):
    if socket.gethostname() != 'thiennh-p6-ohmv-worker-0' or sha(old/'source.json') != prior_sha:
        raise ValueError('Unexpected node or ancestor source')
    if sha(handoff) != handoff_sha:
        raise ValueError('Handoff receipt mismatch')
    prior = json.loads((old/'source.json').read_text())
    pins = json.loads((extra/'pins.json').read_text())
    if set(pins) != set(added) or set(pins) & set(prior) != set(replaced):
        raise ValueError('Invalid registered delta (pins must equal --added; overlap must equal --replaced)')
    if shutil.disk_usage(ROOT).free < 100 * 1024**3:
        raise ValueError('Insufficient disk space')
    base.mkdir(exist_ok=False)
    execution = base/'execution'
    frozen = execution/'source'
    for name, expected in (prior | pins).items():
        src = (extra if name in pins else old/'source')/name
        if sha(src) != expected:
            raise ValueError('Source changed: '+name)
        dst = frozen/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if sha(dst) != expected:
            raise ValueError('Frozen copy mismatch: '+name)
    write_json(execution/'source.json', prior | pins)
    env = ['PYTHONPATH='+str(frozen), 'PYTHONDONTWRITEBYTECODE=1', 'HF_HUB_OFFLINE=1',
           'TRANSFORMERS_OFFLINE=1', 'CUBLAS_WORKSPACE_CONFIG=:4096:8', 'OMP_NUM_THREADS=4',
           'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7']
    worker = [module, 'queue', '--base', str(base),
              '--handoff', str(handoff), '--handoff-sha256', handoff_sha]
    command = 'exec '+shlex.join(['env', *env, sys.executable, '-u', '-m', *worker]) + \
        ' >'+shlex.quote(str(execution/'queue.log'))+' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', base.name, '-c', str(frozen), command], check=True)
    print('LOOP_SCREEN_QUEUE_SUBMITTED', sha(execution/'source.json'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('extra', type=Path)
    p.add_argument('--base', type=Path, default=ROOT/'loop_screen_20261123_a01')
    p.add_argument('--handoff', type=Path, required=True)
    p.add_argument('--handoff-sha256', required=True)
    p.add_argument('--old', type=Path, default=OLD, help='ancestor execution dir whose frozen source is reused')
    p.add_argument('--prior-sha256', default=PRIOR)
    p.add_argument('--added', nargs='+', default=list(ADDED))
    p.add_argument('--module', default='scripts.loop_screen_study')
    p.add_argument('--replaced', nargs='*', default=[], help='prior files deliberately replaced by pins')
    a = p.parse_args()
    launch(a.extra, a.base, a.handoff, a.handoff_sha256, a.old, a.prior_sha256, tuple(a.added), a.module,
           tuple(a.replaced))
