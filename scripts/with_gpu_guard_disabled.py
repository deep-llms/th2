"""Hold the observed controller disable marker while one authorized command runs."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner',required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('Missing child command')
    if socket.gethostname()!='thiennh-p6-tpbw-worker-0':raise ValueError('Unexpected B200 node')
    root=Path('/mnt/local/_gpu_guard');marker=root/'DISABLED'
    source=(root/'gpu_guard.sh').read_text()
    if 'DIS="$DIR/DISABLED"' not in source or 'if [[ -f "$DIS" ]]; then' not in source:
        raise ValueError('Controller marker protocol changed')
    owned=None
    payload=json.dumps({'owner':args.owner,'reason':'User-authorized eight-GPU distillation queue'})+'\n'
    try:
        try:
            with marker.open('x') as f:
                f.write(payload);f.flush();owned=os.fstat(f.fileno())
        except FileExistsError:
            print('PRESERVING_PREEXISTING_GUARD_DISABLE',flush=True)
        print('GUARD_DISABLED_FOR_QUEUE',flush=True)
        result=subprocess.run(command,check=False)
    finally:
        if (owned is not None and marker.exists() and marker.stat().st_ino==owned.st_ino
                and marker.stat().st_dev==owned.st_dev and marker.read_text()==payload):
            marker.unlink()
            print('GUARD_POLICY_RESTORED',flush=True)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
