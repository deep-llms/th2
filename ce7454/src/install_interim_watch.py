"""Install the bounded periodic watcher after its concurrent smoke passed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import py_compile
import shlex
import shutil
import socket
import subprocess
import sys
import time

from attach_band_study import identity, original_running, save
from interim_watch import ORIGINAL, BAND


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if socket.gethostname()!='thiennh-p6-q359-worker-0':raise ValueError('Wrong node')
    smoke=ORIGINAL.parent/'q359-ce7454-interim-smoke-20261016-a01/probe'
    if json.loads((smoke/'released.json').read_text())['status']!='passed':
        raise ValueError('Smoke did not finish successfully')
    for name in ('grip_nodown','band_strong'):
        r=json.loads((smoke/f'cycle_000_{name}/result.json').read_text())
        if r['status']!='passed' or not r['state_unchanged'] or not r['no_gradients']:
            raise ValueError('Smoke invalid')
    lease=json.loads(Path('/mnt/local/_gpu_guard/DISABLED').read_text())
    if lease['output']!=str(ORIGINAL/'supervised'):raise ValueError('Wrong GPU lease')
    owner=identity(lease['owner_pid'])
    if not original_running(ORIGINAL,owner):raise ValueError('Original stopped')
    # Attach well before an original final evaluator can enter its gate.
    if any((ORIGINAL/'supervised/run/study').glob('*_full512')):
        raise ValueError('Too late for this installation protocol')
    oldgate=ORIGINAL/'code/extension_gate.py';cfg=ORIGINAL/'code/extension_gate.json'
    expected='20c45d7c24d98da8a4af465cea954ae28f730462f2bc2c8ec3f1eec5fcbd52e1'
    if sha(oldgate)!=expected:raise ValueError('Gate source changed')
    config=json.loads(cfg.read_text())
    if 'interim_watch' in config:raise ValueError('Already installed')
    a.output.mkdir(parents=True,exist_ok=False)
    code=a.output/'code';code.mkdir()
    for file in Path(__file__).resolve().parent.glob('*.py'):shutil.copy2(file,code/file.name)
    save(a.output/'source_manifest.json',{x.name:sha(x) for x in code.glob('*.py')})
    backup=oldgate.with_suffix('.pre_interim.py')
    if backup.exists():raise FileExistsError(backup)
    shutil.copy2(oldgate,backup)
    # Freeze installed gate but leave optional config disabled until watcher is alive.
    tmp=oldgate.with_suffix('.interim.tmp');shutil.copy2(code/'extension_gate.py',tmp)
    py_compile.compile(str(tmp),doraise=True);tmp.replace(oldgate)
    session=a.output.name
    cmd=shlex.join([sys.executable,'-u',str(code/'interim_watch.py'),'--output',str(a.output/'watch')])
    cmd+=' >'+shlex.quote(str(a.output/'watch.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,cmd],check=True)
    try:
        for _ in range(100):
            if (a.output/'watch/owner.json').exists():break
            time.sleep(.1)
        else:raise RuntimeError('Watcher did not start')
        if not original_running(ORIGINAL,owner):raise RuntimeError('Original stopped during install')
        config['interim_watch']=str(a.output/'watch')
        save(cfg,config)
    except BaseException:
        watch=a.output/'watch'
        if watch.exists():
            (watch/'STOP').touch()
        raise
    receipt={'status':'passed','session':session,'watcher':json.loads((a.output/'watch/owner.json').read_text()),
             'old_gate_sha256':expected,'new_gate_sha256':sha(oldgate),'interval_seconds':1800,
             'gpu':0,'batch_size':8,'training_stopped':False,'original_burn_handoff_preserved':True}
    save(a.output/'installed.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
